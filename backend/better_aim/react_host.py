import asyncio
import json
import os
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from better_aim.agent import create_llm_agent
from better_aim.adjustable_session_service import pop_event
from better_aim.tool_modify_guardrail import zip_tool_schema, extract_arguments_from_schema
from better_aim.utils import generate_random_string, hash_dict
from better_aim.load_mcp_tools import get_mcp_server_tools
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# 全局状态管理 (保持与原main.py兼容)
active_agents: Dict[str, LlmAgent] = {}
history_pool: Dict[str, List[List[str]]] = {}
session_service = InMemorySessionService()

# MCP工具拦截相关状态
pending_events: Dict[str, asyncio.Event] = {}
unmodified_schema_store: Dict[str, Dict] = {}
modified_schema_store: Dict[str, Dict] = {}
modified_args_store: Dict[str, Dict] = {}

# 配置信息
target_tools: List[str] = []
tools_info: List[Dict[str, Any]] = {}
agent_info: Dict[str, Any] = {}
model_config: Dict[str, Any] = {}
mcp_server_url: str = ""
work_path: str = "/tmp"

# FastAPI应用
app = FastAPI(title="Better AIM React API", version="1.0.0")

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有源
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()


# Pydantic模型
class LoginRequest(BaseModel):
    session_id: str

class ChatMessage(BaseModel):
    message: str
    session_id: str

class ModifyParamsRequest(BaseModel):
    session_id: str
    modified_schema: Dict[str, Any]


# 辅助函数
def get_chat_history_file_path(sha_id: str, work_path: str) -> str:
    """获取聊天历史文件路径"""
    history_file_path = os.path.join(work_path, "chat_history")
    os.makedirs(history_file_path, exist_ok=True)
    return os.path.join(history_file_path, f"{sha_id}.json")


def load_chat_history(sha_id: str, work_path: str) -> List[List[str]]:
    """加载聊天历史记录"""
    history_file = get_chat_history_file_path(sha_id, work_path)
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_chat_history(sha_id: str, history: List[List[str]], work_path: str):
    """保存聊天历史记录"""
    history_file = get_chat_history_file_path(sha_id, work_path)
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存聊天历史失败: {e}")


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """与agent异步对话，支持MCP工具拦截"""
    content = types.Content(role='user', parts=[types.Part(text=query)])

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # 处理工具调用
        if event.content and event.content.parts:
            calls = event.get_function_calls()
            if calls:
                for call in calls:
                    tool_name = call.name
                    arguments = call.args

                    # 检查是否需要拦截
                    if tool_name in target_tools:
                        schema = zip_tool_schema(
                            tool_name=tool_name,
                            arguments=arguments,
                            tools_dict=tools_info
                        )

                        # 存储schema并等待用户修改
                        unmodified_schema_store[session_id] = schema
                        pending_events[session_id] = asyncio.Event()

                        # 通知前端需要修改参数
                        await manager.send_message(session_id, {
                            "type": "tool_modify_required",
                            "schema": schema,
                            "tool_name": tool_name
                        })

                        # 等待用户修改完成
                        await pending_events[session_id].wait()

                        # 使用修改后的参数
                        if session_id in modified_args_store:
                            call.args = modified_args_store[session_id]

                        # 清理状态
                        unmodified_schema_store[session_id] = ""

                continue

        # 处理最终响应
        if event.is_final_response():
            if event.content and event.content.parts:
                yield {
                    "type": "final_response",
                    "content": event.content.parts[0].text,
                    "is_final": True
                }
            break
        else:
            if event.content and event.content.parts:
                yield {
                    "type": "streaming_response",
                    "content": event.content.parts[0].text,
                    "is_final": False
                }


# API端点
@app.post("/api/login")
async def login(request: LoginRequest):
    """处理登录逻辑"""
    session_id = request.session_id
    print(f"收到登录请求，会话ID: {session_id}")

    if not session_id:
        raise HTTPException(status_code=400, detail="请填写会话ID")
    elif len(session_id) != 32:
        raise HTTPException(status_code=400, detail="会话ID需要为长度为32的任意字符")

    # 创建或获取agent
    if session_id not in active_agents:
        try:
            agent = create_llm_agent(
                session_id=session_id,
                mcp_tools_url=mcp_server_url,
                agent_info=agent_info,
                model_config=model_config
            )
            active_agents[session_id] = agent
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建Agent失败: {str(e)}")

    # 初始化聊天历史
    if session_id not in history_pool:
        history_pool[session_id] = load_chat_history(session_id, work_path)

    print(f"登录成功，会话ID: {session_id}")
    return {"message": "登录成功", "session_id": session_id}


@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    """与agent对话的HTTP端点 (非流式)"""
    session_id = message.session_id
    user_message = message.message

    if session_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent未找到，请重新登录")

    agent = active_agents[session_id]
    session = await session_service.create_session(
        app_name=agent_info["name"],
        user_id=session_id[:4],
        session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=agent_info["name"],
        session_service=session_service
    )

    full_response = ""
    async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
        full_response += response.get("content", "")

    # 更新聊天历史
    history = history_pool[session_id]
    history.append([user_message, full_response])
    save_chat_history(session_id, history, work_path)

    return {"response": full_response, "is_final": True}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天端点，支持流式响应"""
    await manager.connect(websocket, session_id)

    try:
        if session_id not in active_agents:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Agent未找到，请重新登录"
            }))
            return

        agent = active_agents[session_id]
        session = await session_service.create_session(
            app_name=agent_info["name"],
            user_id=session_id[:4],
            session_id=session_id
        )

        runner = Runner(
            agent=agent,
            app_name=agent_info["name"],
            session_service=session_service
        )

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")

            if not user_message.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "消息不能为空"
                }))
                continue

            response_text = ""
            async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
                if response["type"] in ["streaming_response", "final_response"]:
                    response_text += (response.get("content") or "")
                    await websocket.send_text(json.dumps(response))

            # 更新聊天历史
            history = history_pool[session_id]
            history.append([user_message, response_text])
            save_chat_history(session_id, history, work_path)

    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.post("/api/modify-params")
async def modify_parameters(request: ModifyParamsRequest):
    """处理参数修改请求"""
    session_id = request.session_id
    modified_schema = request.modified_schema

    # 提取修改后的参数
    modified_args = extract_arguments_from_schema(modified_schema)
    modified_args_store[session_id] = modified_args
    modified_schema_store[session_id] = modified_schema

    # 恢复agent执行
    if session_id in pending_events:
        pending_events[session_id].set()

    return {"message": "参数已更新", "modified_args": modified_args}


@app.get("/api/files/{session_id}")
async def list_files(session_id: str):
    """获取会话文件列表"""
    session_dir = os.path.join(work_path, session_id)
    os.makedirs(session_dir, exist_ok=True)

    files = []
    for filename in os.listdir(session_dir):
        file_path = os.path.join(session_dir, filename)
        if os.path.isfile(file_path):
            files.append({
                "name": filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })

    return {"files": sorted(files, key=lambda x: x["name"])}


@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, files: List[UploadFile] = File(...)):
    """上传文件到会话目录"""
    session_dir = os.path.join(work_path, session_id)
    os.makedirs(session_dir, exist_ok=True)

    uploaded_files = []
    for file in files:
        file_path = os.path.join(session_dir, file.filename)

        # 检查文件大小 (10MB限制)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            continue

        with open(file_path, "wb") as f:
            f.write(content)

        uploaded_files.append({
            "name": file.filename,
            "path": file_path,
            "size": len(content)
        })

    return {"uploaded_files": uploaded_files}


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """下载文件"""
    file_path = os.path.join(work_path, session_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(file_path, filename=filename)


@app.get("/api/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """获取聊天历史"""
    history = history_pool.get(session_id, load_chat_history(session_id, work_path))
    return {"history": history}


@app.post("/api/sessions/{session_id}/clear")
async def clear_chat_history(session_id: str):
    """清空聊天历史"""
    history_pool[session_id] = []
    save_chat_history(session_id, [], work_path)
    return {"message": "聊天历史已清空"}


@app.get("/api/schema/{session_id}")
async def get_current_schema(session_id: str):
    """获取当前需要修改的参数schema"""
    schema = unmodified_schema_store.get(session_id, {})
    return {"schema": schema}


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "message": "Backend is running"}

@app.get("/api/config")
async def get_config():
    """获取应用配置信息"""
    print("收到配置请求")
    config = {
        "agent_info": agent_info,
        "mcp_server_url": mcp_server_url,
        "target_tools": target_tools
    }
    print(f"返回配置信息: {config}")
    return config


# 初始化函数
def initialize_server(
    agent_info_dict: Dict[str, Any],
    model_config_dict: Dict[str, Any],
    mcp_url: str,
    work_dir: str = "/tmp",
    tools_modify: List[str] = None
):
    """初始化服务器配置"""
    global agent_info, model_config, mcp_server_url, work_path, target_tools, tools_info

    agent_info = agent_info_dict
    model_config = model_config_dict
    mcp_server_url = mcp_url
    work_path = work_dir
    target_tools = tools_modify or []

    # 加载MCP工具信息
    try:
        tools_info = asyncio.run(get_mcp_server_tools(mcp_server_url))
        print(f"✅ 成功加载 {len(tools_info)} 个MCP工具")
    except Exception as e:
        print(f"⚠️  加载MCP工具失败: {e}")
        tools_info = []


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()