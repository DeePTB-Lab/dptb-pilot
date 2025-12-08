import asyncio
import json
import os
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from dptb_pilot.core.agent import create_llm_agent
from dptb_pilot.core.session import pop_event
from dptb_pilot.core.guardrail import zip_tool_schema, extract_arguments_from_schema
from dptb_pilot.core.utils import generate_random_string, hash_dict
from dptb_pilot.tools.loader import get_mcp_server_tools # Note: loader doesn't exist yet, need to create or fix path
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
    chat_id: Optional[str] = None

class ModifyParamsRequest(BaseModel):
    session_id: str
    modified_schema: Dict[str, Any]


# 辅助函数 - 历史记录管理已合并到 sessions.json
# def get_chat_history_file_path... (removed)
# def load_chat_history... (removed)
# def save_chat_history... (removed)


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
    # 初始化聊天历史 - 已移除，现在按需加载 (lazy load by chat_id)
    # if session_id not in history_pool:
    #     history_pool[session_id] = load_chat_history(session_id, work_path)

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

    chat_id = message.chat_id
    
    # 确保 chat_id 存在
    if not chat_id:
        chat_id = session_id
        print(f"WARNING: No chat_id provided in HTTP request, falling back to user_id: {chat_id}")

    # 懒加载聊天历史
    if chat_id not in history_pool:
        history_pool[chat_id] = load_session_history(session_id, chat_id, work_path)

    # 更新聊天历史
    history = history_pool[chat_id]
    history.append([user_message, full_response])
    
    # 同步更新 sessions.json
    update_session_history(session_id, chat_id, history, work_path)

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
        
        try:
            session = await session_service.create_session(
                app_name=agent_info["name"],
                user_id=session_id[:4],
                session_id=session_id
            )
        except Exception as e:
            print(f"ERROR: Failed to create session: {e}")
            raise

        runner = Runner(
            agent=agent,
            app_name=agent_info["name"],
            session_service=session_service
        )

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            chat_id = message_data.get("chat_id")

            if not user_message.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "消息不能为空"
                }))
                continue

            response_text = ""
            try:
                async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
                    if response["type"] in ["streaming_response", "final_response"]:
                        response_text += (response.get("content") or "")
                        
                        # Check for usage metadata
                        if "usage" in response:
                             # Send usage info to frontend
                             await websocket.send_text(json.dumps({
                                 "type": "usage_update",
                                 "usage": response["usage"]
                             }))
                             
                        await websocket.send_text(json.dumps(response))
            except Exception as e:
                print(f"Error during agent execution: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to extract more info if it's an ExceptionGroup (Python 3.11+)
                if hasattr(e, 'exceptions'):
                    for i, exc in enumerate(e.exceptions):
                        print(f"Sub-exception {i+1}: {exc}")
                        
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Agent execution error: {str(e)}. Please try again."
                }))
                continue

            # 确保 chat_id 存在
            if not chat_id:
                # 如果没有 chat_id，尝试使用 session_id (兼容旧逻辑，但不推荐)
                chat_id = session_id
                print(f"WARNING: No chat_id provided, falling back to user_id: {chat_id}")

            # 懒加载聊天历史 (从 sessions.json)
            if chat_id not in history_pool:
                history_pool[chat_id] = load_session_history(session_id, chat_id, work_path)

            # 更新聊天历史
            history = history_pool[chat_id]
            history.append([user_message, response_text])
            
            # 同步更新 sessions.json (这是唯一的持久化存储)
            update_session_history(session_id, chat_id, history, work_path)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        print(f"CRITICAL ERROR in websocket_chat: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011) # Internal Error
        except:
            pass
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
    session_dir = os.path.join(work_path, session_id, "files")
    print(f"Listing files from: {session_dir}")
    os.makedirs(session_dir, exist_ok=True)

    files = []
    if os.path.exists(session_dir):
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                files.append({
                    "name": filename,
                    "path": file_path,
                    "size": stats.st_size,
                    "updated_at": stats.st_mtime
                })
    print(f"Found {len(files)} files")
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, files: List[UploadFile] = File(...)):
    """上传文件到会话目录"""
    session_dir = os.path.join(work_path, session_id, "files")
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
    file_path = os.path.join(work_path, session_id, "files", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(file_path, filename=filename)

@app.delete("/api/files/{session_id}/{filename}")
async def delete_file(session_id: str, filename: str):
    """删除文件"""
    file_path = os.path.join(work_path, session_id, "files", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        os.remove(file_path)
        return {"message": "文件已删除", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")


@app.get("/api/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """获取聊天历史 (Legacy)"""
    history = history_pool.get(session_id, load_chat_history(session_id, work_path))
    return {"history": history}


@app.post("/api/sessions/{session_id}/clear")
async def clear_chat_history(session_id: str):
    """清空聊天历史 (Legacy)"""
    history_pool[session_id] = []
    save_chat_history(session_id, [], work_path)
    return {"message": "聊天历史已清空"}


class SaveSessionsRequest(BaseModel):
    sessions: List[Dict[str, Any]]


@app.get("/api/user/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """获取用户的所有聊天会话"""
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    print(f"Loading sessions for {user_id} from {sessions_file}")
    
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            # 转换历史记录格式以适配前端: [[q, a], ...] -> [{role: user, content: q}, {role: assistant, content: a}, ...]
            for session in sessions:
                raw_history = session.get("history", [])
                formatted_history = []
                for item in raw_history:
                    if isinstance(item, list) and len(item) >= 2:
                        formatted_history.append({"role": "user", "content": item[0]})
                        formatted_history.append({"role": "assistant", "content": item[1]})
                session["history"] = formatted_history
                # Update message count to reflect total messages (user + assistant)
                session["message_count"] = len(formatted_history)
            
            print(f"Loaded {len(sessions)} sessions")
            return {"sessions": sessions}
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return {"sessions": []}
    print("Sessions file not found")
    return {"sessions": []}


def load_session_history(user_id: str, chat_id: str, work_path: str) -> List[List[str]]:
    """从 sessions.json 加载特定会话的历史记录"""
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    
    if not os.path.exists(sessions_file):
        return []

    try:
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        for session in sessions:
            if session.get("chat_id") == chat_id:
                return session.get("history", [])
    except Exception as e:
        print(f"Error loading session history: {e}")
    
    return []


def update_session_history(user_id: str, session_id: str, history: List[List[str]], work_path: str):
    """更新用户会话列表中的历史记录"""
    print(f"DEBUG: Updating session history for User: {user_id}, Chat: {session_id}, History Len: {len(history)}")
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    
    if not os.path.exists(sessions_file):
        print(f"DEBUG: Sessions file not found: {sessions_file}")
        return

    try:
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        updated = False
        for session in sessions:
            print(f"DEBUG: Checking session {session.get('chat_id')} against {session_id}")
            if session.get("chat_id") == session_id:
                session["history"] = history
                session["last_active"] = datetime.now().isoformat()
                session["message_count"] = len(history)
                updated = True
                print(f"DEBUG: Found and updated session {session_id}")
                break
        
        if updated:
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            print("DEBUG: Successfully saved sessions.json")
        else:
            print(f"DEBUG: Chat ID {session_id} not found in sessions.json")
            
    except Exception as e:
        print(f"Failed to update session history in sessions.json: {e}")


@app.post("/api/user/{user_id}/sessions")
async def save_user_sessions(user_id: str, request: SaveSessionsRequest):
    """保存用户的所有聊天会话"""
    user_dir = os.path.join(work_path, user_id)
    os.makedirs(user_dir, exist_ok=True)
    sessions_file = os.path.join(user_dir, "sessions.json")
    print(f"Saving {len(request.sessions)} sessions for {user_id} to {sessions_file}")
    
    try:
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(request.sessions, f, ensure_ascii=False, indent=2)
        return {"message": "Sessions saved successfully"}
    except Exception as e:
        print(f"Failed to save sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save sessions: {str(e)}")


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