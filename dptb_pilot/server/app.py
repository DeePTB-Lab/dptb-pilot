import asyncio
import json
import os
import uuid
import copy
import re
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
from litellm import acompletion
from dptb_pilot.core.logger import get_logger
from dptb_pilot.core.photon_service import get_photon_service, PhotonChargeResult
from dptb_pilot.core.photon_config import CHARGING_ENABLED

logger = get_logger(__name__)

BACKEND_BUILD_ID = "2026-07-29-execution-config-v1"
SERVER_CAPABILITIES = ["execution_config", "remote_executor_binding"]


# 全局状态管理 (保持与原main.py兼容)
active_agents: Dict[str, LlmAgent] = {}
session_execution_configs: Dict[str, Dict[str, Any]] = {}
history_pool: Dict[str, List[List[str]]] = {}
session_service = InMemorySessionService()

# MCP工具拦截相关状态
pending_events: Dict[str, asyncio.Event] = {}
unmodified_schema_store: Dict[str, Dict] = {}
modified_schema_store: Dict[str, Dict] = {}
modified_args_store: Dict[str, Dict] = {}

# Workflow planning approval state.  Complex tasks pause here before tool execution.
workflow_plan_store: Dict[str, Dict[str, Any]] = {}
workflow_plan_events: Dict[str, asyncio.Event] = {}
workflow_plan_decisions: Dict[str, str] = {}
workflow_plan_tasks: Dict[str, str] = {}

# 终止执行相关状态
cancel_execution_events: Dict[str, asyncio.Event] = {}
termination_requested: Dict[str, bool] = {}

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

class ExecutionConfigRequest(BaseModel):
    session_id: str
    execution_mode: str = 'Local'
    selected_machine_id: Optional[str] = None
    remote_machine: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    message: str
    session_id: str
    chat_id: Optional[str] = None

class ModifyParamsRequest(BaseModel):
    session_id: str
    modified_schema: Dict[str, Any]
    execution_mode: str = 'Local'
    selected_machine_id: Optional[str] = None
    remote_machine: Optional[Dict[str, Any]] = None  # 包含完整的远程机器配置

class TerminateExecutionRequest(BaseModel):
    session_id: str

class WorkflowPlanDecisionRequest(BaseModel):
    session_id: str
    decision: str


def build_execution_config(
    execution_mode: str,
    remote_machine: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if execution_mode != 'Remote':
        return {}
    if not remote_machine:
        raise ValueError("Remote execution requires a selected machine")

    machine_type = str(remote_machine.get('type') or '').strip().lower()
    config = remote_machine.get('config') or {}
    if not isinstance(config, dict):
        raise ValueError("Remote machine config must be an object")

    if machine_type == 'bohrium':
        username = str(config.get('username') or '').strip()
        password = str(config.get('password') or '')
        project_id = str(config.get('project_id') or '').strip()
        if not username or not password or not project_id:
            raise ValueError(
                "Bohrium username, password, and project_id are required"
            )
        try:
            program_id = int(project_id)
        except ValueError as exc:
            raise ValueError("Bohrium project_id must be an integer") from exc

        return {
            'executor': {
                'type': 'dispatcher',
                'machine': {
                    'batch_type': 'Bohrium',
                    'context_type': 'Bohrium',
                    'remote_profile': {
                        'email': username,
                        'password': password,
                        'program_id': program_id,
                        'input_data': {
                            'image_name': config.get('image_name') or
                            'registry.dp.tech/dptech/dp/native/prod-35271/dptb-pilot-test:0.2',
                            'job_type': 'container',
                            'platform': 'ali',
                            'scass_type': config.get('scass_type') or 'c2_m4_cpu',
                        },
                    },
                },
            },
            'storage': {
                'type': 'bohrium',
                'username': username,
                'password': password,
                'project_id': project_id,
            },
        }

    if machine_type == 'slurm':
        required = ['remote_root', 'hostname', 'username', 'key_filename']
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError(f"Slurm config is missing: {', '.join(missing)}")
        return {
            'executor': {
                'type': 'dispatcher',
                'machine': {
                    'batch_type': 'Slurm',
                    'context_type': 'SSHContext',
                    'local_root': './',
                    'remote_root': config['remote_root'],
                    'remote_profile': {
                        'hostname': config['hostname'],
                        'username': config['username'],
                        'timeout': 600,
                        'port': int(config.get('port', 22)),
                        'key_filename': config['key_filename'],
                    },
                },
                'resources': {
                    'number_node': int(config.get('number_node', 1)),
                    'gpu_per_node': int(config.get('gpu_per_node') or 0),
                    'cpu_per_node': int(config.get('cpu_per_node') or 1),
                    'queue_name': config.get('queue_name'),
                    'custom_flags': [config.get('custom_flags', ''), ''],
                    'source_list': [],
                    'module_list': [],
                },
            },
        }

    raise ValueError(
        f"Unsupported remote machine type: {remote_machine.get('type')}"
    )


def create_session_agent(session_id: str) -> LlmAgent:
    return create_llm_agent(
        session_id=session_id,
        mcp_tools_url=mcp_server_url,
        agent_info=agent_info,
        model_config=model_config,
        execution_config=session_execution_configs.get(session_id),
    )


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ('***' if key.lower() in {'password', 'authorization', 'token'}
                  else redact_secrets(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def generate_executor_and_storage(
    execution_mode: str,
    remote_machine: Optional[Dict[str, Any]],
    tool_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据执行模式和远程机器配置自动生成 Executor 和 Storage 参数

    Args:
        execution_mode: 执行模式 ('Local' 或 'Remote')
        remote_machine: 远程机器配置
        tool_schema: 工具 schema

    Returns:
        更新后的工具 schema，包含自动生成的 Executor 和 Storage 参数
    """
    if execution_mode != 'Remote':
        return tool_schema

    try:
        execution_config = build_execution_config(execution_mode, remote_machine)
    except ValueError as exc:
        logger.warning(f"[AutoGenerate] 无效的远程机器配置: {exc}")
        return tool_schema

    machine_type = remote_machine.get('type')
    config = remote_machine.get('config', {})

    logger.info("=" * 80)
    logger.info(f"[AutoGenerate] 开始自动生成 Executor 和 Storage 参数")
    logger.info(f"[AutoGenerate] 机器类型: {machine_type}")
    logger.info(f"[AutoGenerate] 机器配置: {json.dumps(redact_secrets(config), ensure_ascii=False, indent=2)}")

    # 深拷贝 tool_schema 避免修改原对象
    result_schema = copy.deepcopy(tool_schema)

    # 更新 schema 中的 Executor 和 Storage 参数
    properties = result_schema.get('input_schema', {}).get('properties', {})

    # 生成 Executor 配置（支持大写和小写的 key）
    executor_key = None
    for key in ['Executor', 'executor']:
        if key in properties:
            executor_key = key
            break

    if executor_key:
        executor_config = execution_config['executor']
        logger.info(f"[AutoGenerate] {machine_type} Executor 配置已生成")
        logger.info(f"[AutoGenerate] Executor: {json.dumps(redact_secrets(executor_config), ensure_ascii=False, indent=2)}")
        properties[executor_key]['user_input'] = executor_config

    # 生成 Storage 配置（仅 Bohrium 类型）
    storage_key = None
    for key in ['Storage', 'storage']:
        if key in properties:
            storage_key = key
            break

    if storage_key and machine_type == 'Bohrium':
        storage_config = execution_config['storage']
        logger.info(f"[AutoGenerate] Bohrium Storage 配置已生成")
        logger.info(f"[AutoGenerate] Storage: {json.dumps(redact_secrets(storage_config), ensure_ascii=False, indent=2)}")
        properties[storage_key]['user_input'] = storage_config

    logger.info(f"[AutoGenerate] 完成 Executor 和 Storage 参数自动生成")
    logger.info("=" * 80)

    return result_schema


def should_plan_workflow(query: str) -> bool:
    """Return True when a user request likely needs multi-step execution."""
    text = (query or "").strip()
    if len(text) >= 80:
        return True

    complex_markers = [
        "workflow", "plan", "steps", "pipeline", "calculate", "scan", "compare",
        "train", "optimize", "submit", "batch", "analyze", "download", "generate",
        "工作流", "计划", "流程", "步骤", "计算", "扫描", "比较", "训练", "优化",
        "提交", "批量", "分析", "下载", "生成", "拟合", "能带", "结构", "材料",
    ]
    hits = sum(1 for marker in complex_markers if marker.lower() in text.lower())
    return hits >= 2


def _is_workflow_approval(query: str) -> bool:
    text = (query or "").strip().lower()
    return text in {"approve", "approved", "yes", "ok", "go", "run", "execute", "同意", "通过", "确认", "执行", "开始"}


def _is_workflow_rejection(query: str) -> bool:
    text = (query or "").strip().lower()
    return text in {"reject", "rejected", "no", "cancel", "stop", "拒绝", "取消", "不通过", "停止"}


def _tool_catalog(limit: int = 60) -> List[Dict[str, str]]:
    catalog = []
    for tool in tools_info[:limit]:
        catalog.append({
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", ""))[:500],
        })
    return catalog


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("empty response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _normalize_workflow_plan(plan: Dict[str, Any], user_task: str) -> Dict[str, Any]:
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    normalized_steps = []
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        tools = step.get("tools") or step.get("modules") or []
        if isinstance(tools, str):
            tools = [tools]
        depends_on = step.get("depends_on") or []
        if isinstance(depends_on, (str, int)):
            depends_on = [depends_on]
        normalized_steps.append({
            "id": str(step.get("id") or f"S{idx}"),
            "title": str(step.get("title") or step.get("name") or f"Step {idx}"),
            "purpose": str(step.get("purpose") or step.get("description") or ""),
            "tools": [str(item) for item in tools],
            "inputs": step.get("inputs") if isinstance(step.get("inputs"), list) else [],
            "outputs": step.get("outputs") if isinstance(step.get("outputs"), list) else [],
            "depends_on": [str(item) for item in depends_on],
        })

    if not normalized_steps:
        return _fallback_workflow_plan(user_task)

    return {
        "title": str(plan.get("title") or "Workflow plan"),
        "summary": str(plan.get("summary") or "A reviewed execution plan for the requested task."),
        "task": user_task,
        "steps": normalized_steps,
        "risks": plan.get("risks") if isinstance(plan.get("risks"), list) else [],
        "questions": plan.get("questions") if isinstance(plan.get("questions"), list) else [],
        "review": plan.get("review") if isinstance(plan.get("review"), dict) else None,
    }


def _fallback_workflow_plan(user_task: str) -> Dict[str, Any]:
    available = [tool["name"] for tool in _tool_catalog(limit=20) if tool["name"]]
    return {
        "title": "Workflow plan",
        "summary": "Pilot will inspect available files and knowledge first, then run the required DeePTB/materials tools after your approval.",
        "task": user_task,
        "steps": [
            {
                "id": "S1",
                "title": "Clarify inputs and constraints",
                "purpose": "Identify uploaded structures, target properties, model files, execution mode, and missing assumptions.",
                "tools": ["list_workspace_files", "search_knowledge_base"],
                "inputs": ["user request", "session files"],
                "outputs": ["resolved task inputs"],
                "depends_on": [],
            },
            {
                "id": "S2",
                "title": "Prepare calculation assets",
                "purpose": "Generate or select the required DeePTB configuration, structure files, and runtime parameters.",
                "tools": available[:6],
                "inputs": ["resolved task inputs"],
                "outputs": ["ready-to-run files and parameters"],
                "depends_on": ["S1"],
            },
            {
                "id": "S3",
                "title": "Execute approved modules",
                "purpose": "Run the selected local or remote computational modules with reviewed parameters.",
                "tools": available[6:12],
                "inputs": ["ready-to-run files and parameters"],
                "outputs": ["raw calculation outputs"],
                "depends_on": ["S2"],
            },
            {
                "id": "S4",
                "title": "Postprocess and report",
                "purpose": "Collect generated artifacts, visualize results when possible, and summarize conclusions.",
                "tools": ["download_artifact", "visualize_structure", "visualize_brillouin_zone"],
                "inputs": ["raw calculation outputs"],
                "outputs": ["figures, files, and final answer"],
                "depends_on": ["S3"],
            },
        ],
        "risks": ["Some tool parameters may still require confirmation during execution."],
        "questions": [],
        "review": None,
    }


def _normalize_workflow_review(review: Dict[str, Any]) -> Dict[str, Any]:
    def list_of_strings(key: str) -> List[str]:
        value = review.get(key)
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    verdict = str(review.get("verdict") or "needs_attention").lower().strip()
    if verdict not in {"pass", "needs_attention", "block"}:
        verdict = "needs_attention"

    return {
        "verdict": verdict,
        "summary": str(review.get("summary") or "Pilot reviewed the workflow plan against the DeePTB knowledge base."),
        "scientific_logic": list_of_strings("scientific_logic"),
        "execution_logic": list_of_strings("execution_logic"),
        "blocking_issues": list_of_strings("blocking_issues"),
        "warnings": list_of_strings("warnings"),
        "recommendations": list_of_strings("recommendations"),
        "evidence": list_of_strings("evidence"),
        "reviewed_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


async def generate_workflow_review(plan: Dict[str, Any]) -> Dict[str, Any]:
    task = str(plan.get("task") or "")
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    tool_names = sorted({
        str(tool)
        for step in steps
        for tool in (step.get("tools") or [])
        if str(tool).strip()
    })
    step_titles = "; ".join(str(step.get("title", "")) for step in steps)
    kb_queries = [
        f"DeePTB workflow scientific validity {task} {step_titles}",
        f"DeePTB tools runtime logic inputs outputs {' '.join(tool_names)}",
        f"DeePTB common mistakes configuration calculation workflow {task}",
    ]

    evidence_chunks: List[str] = []
    try:
        from dptb_pilot.tools.modules.knowledge.rag_tool import search_knowledge_base

        for query in kb_queries:
            result = search_knowledge_base(query=query[:500], n_results=3)
            evidence_chunks.append(result[:3500])
    except Exception as exc:
        logger.warning(f"[WorkflowReview] Knowledge search failed, reviewing with planner context only: {exc}")
        evidence_chunks.append(f"Knowledge search unavailable: {exc}")

    system = {
        "role": "system",
        "content": (
            "You are DeePTB Pilot's workflow reviewer. Check both scientific logic and execution logic. "
            "Use the supplied knowledge-base evidence as the main source of truth. "
            "Return strict JSON only with keys: verdict, summary, scientific_logic, execution_logic, "
            "blocking_issues, warnings, recommendations, evidence. "
            "verdict must be one of pass, needs_attention, block."
        ),
    }
    prompt = {
        "role": "user",
        "content": (
            "Review this proposed workflow plan before execution.\n\n"
            f"Workflow plan JSON:\n{json.dumps(plan, ensure_ascii=False)}\n\n"
            f"Knowledge-base evidence:\n{chr(10).join(evidence_chunks)}"
        ),
    }

    try:
        response = await acompletion(messages=[system, prompt], **model_config)
        content = response.choices[0].message.content
        return _normalize_workflow_review(_extract_json_object(content))
    except Exception as exc:
        logger.warning(f"[WorkflowReview] LLM review failed, using fallback report: {exc}")
        return _normalize_workflow_review({
            "verdict": "needs_attention",
            "summary": "Pilot could not complete the model-based review. The plan can still be manually reviewed before approval.",
            "scientific_logic": [],
            "execution_logic": [],
            "blocking_issues": [],
            "warnings": [f"Automatic review failed: {exc}"],
            "recommendations": ["Manually verify the task assumptions, required inputs, selected tools, and output dependencies before approving."],
            "evidence": evidence_chunks[:3],
        })


async def generate_workflow_plan(user_task: str) -> Dict[str, Any]:
    prompt = {
        "role": "user",
        "content": (
            "Design a workflow plan for the user task using only the available DeePTB Pilot tools. "
            "Return strict JSON with keys: title, summary, steps, risks, questions. "
            "Each step must have id, title, purpose, tools, inputs, outputs, depends_on. "
            "Do not execute tools. User task:\n"
            f"{user_task}\n\nAvailable tools:\n{json.dumps(_tool_catalog(), ensure_ascii=False)}"
        ),
    }
    system = {
        "role": "system",
        "content": (
            "You are DeePTB Pilot's workflow planner. Produce concise, auditable plans. "
            "Use tool names exactly when relevant. Prefer 3-6 steps. Return JSON only."
        ),
    }

    try:
        response = await acompletion(messages=[system, prompt], **model_config)
        content = response.choices[0].message.content
        return _normalize_workflow_plan(_extract_json_object(content), user_task)
    except Exception as exc:
        logger.warning(f"[WorkflowPlan] LLM planning failed, using fallback plan: {exc}")
        return _fallback_workflow_plan(user_task)


def render_workflow_plan_markdown(plan: Dict[str, Any]) -> str:
    lines = [
        "## Workflow plan for review",
        "",
        f"**Goal:** {plan.get('summary', '')}",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    steps = plan.get("steps", [])
    for step in steps:
        label = f"{step.get('id')}: {step.get('title')}"
        lines.append(f"  {step.get('id')}[\"{label}\"]")
    for step in steps:
        for dep in step.get("depends_on", []):
            lines.append(f"  {dep} --> {step.get('id')}")
    lines.extend(["```", "", "| Step | Purpose | Modules |", "| --- | --- | --- |"])
    for step in steps:
        tools = ", ".join(step.get("tools") or ["none"])
        lines.append(f"| {step.get('id')} {step.get('title')} | {step.get('purpose')} | `{tools}` |")
    if plan.get("risks"):
        lines.extend(["", "**Risks / checks:**"])
        lines.extend([f"- {item}" for item in plan["risks"]])
    if plan.get("questions"):
        lines.extend(["", "**Open questions:**"])
        lines.extend([f"- {item}" for item in plan["questions"]])
    review = plan.get("review") if isinstance(plan.get("review"), dict) else None
    if review:
        lines.extend(["", "## Pilot review report", "", f"**Verdict:** `{review.get('verdict', 'needs_attention')}`", "", str(review.get("summary", ""))])
        for title, key in [
            ("Scientific logic", "scientific_logic"),
            ("Execution logic", "execution_logic"),
            ("Blocking issues", "blocking_issues"),
            ("Warnings", "warnings"),
            ("Recommendations", "recommendations"),
        ]:
            items = review.get(key) if isinstance(review.get(key), list) else []
            if items:
                lines.extend(["", f"**{title}:**"])
                lines.extend([f"- {item}" for item in items])
    lines.append("\nUse the right panel to review manually, ask Pilot to review, approve, or reject this workflow.")
    return "\n".join(lines)


def build_approved_workflow_query(user_task: str, plan: Dict[str, Any]) -> str:
    return (
        "The user has approved this workflow plan. Execute the original task following it. "
        "If a later tool call needs parameter confirmation, pause for the existing parameter review UI.\n\n"
        f"Original task:\n{user_task}\n\nApproved workflow JSON:\n"
        f"{json.dumps(plan, ensure_ascii=False)}"
    )

async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """与agent异步对话，支持MCP工具拦截"""
    content = types.Content(role='user', parts=[types.Part(text=query)])

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # 检查是否被终止
        if termination_requested.get(session_id, False):
            logger.info(f"[CallAgent] 会话 {session_id} 已请求终止")
            yield {
                "type": "final_response",
                "content": "执行已终止",
                "is_final": True
            }
            # 清理终止状态
            termination_requested[session_id] = False
            pending_event = pending_events.get(session_id)
            if isinstance(pending_event, asyncio.Event):
                pending_event.set()
            else:
                pending_events.pop(session_id, None)
            break

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
                        pending_event = asyncio.Event()
                        pending_events[session_id] = pending_event
                        cancel_event = asyncio.Event()
                        cancel_execution_events[session_id] = cancel_event

                        # 通知前端需要修改参数
                        await manager.send_message(session_id, {
                            "type": "tool_modify_required",
                            "schema": schema,
                            "tool_name": tool_name
                        })

                        # 终止接口也会触发 pending_event，使等待立即返回。
                        try:
                            await asyncio.wait_for(pending_event.wait(), timeout=600.0)
                        except asyncio.TimeoutError:
                            logger.warning(f"[CallAgent] 会话 {session_id} 等待参数修改超时")
                            break
                        finally:
                            if pending_events.get(session_id) is pending_event:
                                pending_events.pop(session_id, None)
                            if cancel_execution_events.get(session_id) is cancel_event:
                                cancel_execution_events.pop(session_id, None)

                        # 检查是否被终止
                        if termination_requested.get(session_id, False):
                            logger.info(f"[CallAgent] 会话 {session_id} 在参数修改阶段被终止")
                            yield {
                                "type": "final_response",
                                "content": "执行已终止",
                                "is_final": True
                            }
                            # 清理状态
                            termination_requested[session_id] = False
                            break

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
    logger.info(f"收到登录请求，会话ID: {session_id}")

    if not session_id:
        raise HTTPException(status_code=400, detail="请填写会话ID")
    elif len(session_id) != 32:
        raise HTTPException(status_code=400, detail="会话ID需要为长度为32的任意字符")

    # 创建或获取agent
    if session_id not in active_agents:
        try:
            active_agents[session_id] = create_session_agent(session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建Agent失败: {str(e)}")

    logger.info(f"登录成功，会话ID: {session_id}")
    return {"message": "登录成功", "session_id": session_id}


@app.post("/api/execution-config")
async def update_execution_config(request: ExecutionConfigRequest):
    try:
        execution_config = build_execution_config(
            request.execution_mode,
            request.remote_machine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if execution_config:
        session_execution_configs[request.session_id] = execution_config
    else:
        session_execution_configs.pop(request.session_id, None)

    if request.session_id in active_agents:
        try:
            active_agents[request.session_id] = create_session_agent(
                request.session_id
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to apply execution config: {exc}",
            ) from exc

    return {
        "message": "Execution config applied",
        "execution_mode": request.execution_mode,
        "selected_machine_id": request.selected_machine_id,
        "remote_enabled": bool(execution_config),
    }


@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    """与agent对话的HTTP端点 (非流式)"""
    session_id = message.session_id
    user_message = message.message

    if session_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent未找到，请重新登录")

    agent = active_agents[session_id]
    try:
        session = await session_service.create_session(
            app_name=agent_info["name"],
            user_id=session_id[:4],
            session_id=session_id
        )
    except Exception:
        session = await session_service.get_session(
            app_name=agent_info["name"],
            user_id=session_id[:4],
            session_id=session_id
        )

    runner = Runner(
        agent=agent,
        app_name=agent_info["name"],
        session_service=session_service
    )

    has_approved_workflow = False
    if session_id in workflow_plan_store and _is_workflow_rejection(user_message):
        workflow_plan_store.pop(session_id, None)
        workflow_plan_tasks.pop(session_id, None)
        return {"response": "Workflow was rejected. Send a revised task to create another plan.", "is_final": True}

    if session_id in workflow_plan_store and _is_workflow_approval(user_message):
        plan = workflow_plan_store.pop(session_id)
        original_task = workflow_plan_tasks.pop(session_id, plan.get("task", ""))
        user_message = build_approved_workflow_query(original_task, plan)
        has_approved_workflow = True

    if not has_approved_workflow and should_plan_workflow(user_message):
        plan = await generate_workflow_plan(user_message)
        workflow_plan_store[session_id] = plan
        workflow_plan_tasks[session_id] = user_message
        return {
            "response": render_workflow_plan_markdown(plan),
            "is_final": False,
            "requires_workflow_approval": True,
            "workflow_plan": plan,
        }

    full_response = ""
    async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
        full_response += (response.get("content") or "")

    chat_id = message.chat_id
    
    # 确保 chat_id 存在
    if not chat_id:
        chat_id = session_id
        print(f"WARNING: No chat_id provided in HTTP request, falling back to user_id: {chat_id}")
        logger.warning(f"No chat_id provided in HTTP request, falling back to user_id: {chat_id}")

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

    # 获取 cookies 用于光子收费
    cookies = None
    try:
        # FastAPI WebSocket 不直接提供 cookies 属性，需要从请求中获取
        cookies = dict(websocket._cookies) if hasattr(websocket, '_cookies') else {}
        logger.info(f"WebSocket connection cookies: {list(cookies.keys())}")
    except Exception as e:
        logger.warning(f"Failed to get WebSocket cookies: {e}")
        cookies = {}

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
            logger.error(f"Failed to create session: {e}")
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
            original_user_message = user_message
            chat_id = message_data.get("chat_id")

            if not user_message.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "消息不能为空"
                }))
                continue

            response_text = ""
            usage_metadata = None
            try:
                if should_plan_workflow(user_message):
                    plan = await generate_workflow_plan(user_message)
                    plan_markdown = render_workflow_plan_markdown(plan)
                    workflow_plan_store[session_id] = plan
                    workflow_plan_tasks[session_id] = user_message
                    workflow_plan_decisions.pop(session_id, None)
                    workflow_plan_events[session_id] = asyncio.Event()

                    response_text = plan_markdown
                    await websocket.send_text(json.dumps({
                        "type": "streaming_response",
                        "content": response_text,
                        "is_final": False
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "workflow_plan_required",
                        "workflow_plan": plan
                    }))

                    try:
                        await asyncio.wait_for(workflow_plan_events[session_id].wait(), timeout=1800.0)
                    except asyncio.TimeoutError:
                        workflow_plan_store.pop(session_id, None)
                        workflow_plan_tasks.pop(session_id, None)
                        workflow_plan_events.pop(session_id, None)
                        await websocket.send_text(json.dumps({
                            "type": "final_response",
                            "content": response_text + "\n\nWorkflow review timed out. Please send the task again when ready.",
                            "is_final": True
                        }))
                        continue

                    decision = workflow_plan_decisions.pop(session_id, "rejected")
                    workflow_plan_events.pop(session_id, None)
                    if decision != "approved":
                        workflow_plan_store.pop(session_id, None)
                        workflow_plan_tasks.pop(session_id, None)
                        response_text = response_text + "\n\nWorkflow was rejected. Revise the request or send a new task to create another plan."
                        await websocket.send_text(json.dumps({
                            "type": "final_response",
                            "content": response_text,
                            "is_final": True
                        }))
                        continue

                    plan = workflow_plan_store.get(session_id, plan)
                    response_text = render_workflow_plan_markdown(plan)
                    user_message = build_approved_workflow_query(user_message, plan)
                    response_text = response_text + "\n\n---\n\nWorkflow approved. Starting execution...\n\n"
                    await websocket.send_text(json.dumps({
                        "type": "streaming_response",
                        "content": response_text,
                        "is_final": False
                    }))
                    workflow_plan_store.pop(session_id, None)
                    workflow_plan_tasks.pop(session_id, None)

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
                             # Store usage metadata for photon charging
                             usage_metadata = response["usage"]

                        outbound = dict(response)
                        outbound["content"] = response_text
                        await websocket.send_text(json.dumps(outbound))
            except Exception as e:
                logger.error(f"Error during agent execution: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to extract more info if it's an ExceptionGroup (Python 3.11+)
                if hasattr(e, 'exceptions'):
                    for i, exc in enumerate(e.exceptions):
                        logger.error(f"Sub-exception {i+1}: {exc}")
                        
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Agent execution error: {str(e)}. Please try again."
                }))
                continue

            # 执行光子收费（如果启用）
            charge_result = None
            if CHARGING_ENABLED and usage_metadata:
                try:
                    photon_service = get_photon_service()
                    if photon_service:
                        input_tokens = usage_metadata.get("promptTokenCount", 0) or usage_metadata.get("prompt_tokens", 0)
                        output_tokens = usage_metadata.get("candidatesTokenCount", 0) or usage_metadata.get("candidates_tokens", 0)

                        logger.info(f"Processing photon charge - Input: {input_tokens}, Output: {output_tokens}")

                        charge_result = await photon_service.charge_photon(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            tool_calls=0,
                            websocket_cookies=cookies
                        )

                        # 发送收费结果到前端
                        await websocket.send_text(json.dumps({
                            "type": "charge_result",
                            "charge_result": {
                                "success": charge_result.success,
                                "code": charge_result.code,
                                "message": charge_result.message,
                                "biz_no": str(charge_result.biz_no) if charge_result.biz_no else None,
                                "photon_amount": charge_result.photon_amount,
                                "rmb_amount": charge_result.rmb_amount
                            }
                        }))

                        if charge_result.success:
                            logger.info(f"Photon charge successful: {charge_result.message}")
                        else:
                            logger.warning(f"Photon charge failed: {charge_result.message}")
                except Exception as charge_error:
                    logger.error(f"Error during photon charging: {charge_error}")
                    await websocket.send_text(json.dumps({
                        "type": "charge_result",
                        "charge_result": {
                            "success": False,
                            "code": -1,
                            "message": f"收费异常: {str(charge_error)}",
                            "photon_amount": 0,
                            "rmb_amount": 0.0
                        }
                    }))

            # 确保 chat_id 存在
            if not chat_id:
                # 如果没有 chat_id，尝试使用 session_id (兼容旧逻辑，但不推荐)
                chat_id = session_id
                logger.warning(f"No chat_id provided, falling back to user_id: {chat_id}")

            # 懒加载聊天历史 (从 sessions.json)
            if chat_id not in history_pool:
                history_pool[chat_id] = load_session_history(session_id, chat_id, work_path)

            # 更新聊天历史
            history = history_pool[chat_id]
            history.append([original_user_message, response_text])
            
            # 同步更新 sessions.json (这是唯一的持久化存储)
            update_session_history(session_id, chat_id, history, work_path)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in websocket_chat: {e}")
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

    logger.info("=" * 80)
    logger.info(f"[ModifyParams] 收到参数修改请求")
    logger.info(f"[ModifyParams] Session ID: {session_id}")
    logger.info(f"[ModifyParams] 工具名称: {modified_schema.get('name', 'unknown')}")
    logger.info(f"[ModifyParams] 执行模式: {request.execution_mode}")
    logger.info(f"[ModifyParams] 选中的机器ID: {request.selected_machine_id}")
    logger.info(f"[ModifyParams] 修改前的Schema: {json.dumps(redact_secrets(modified_schema), ensure_ascii=False, indent=2)}")
    logger.info("=" * 80)

    # 构建工作目录路径：{work_path}/{session_id}/files（确保是绝对路径）
    session_files_dir = os.path.abspath(os.path.join(work_path, session_id, "files"))

    # 自动生成 Executor 和 Storage 参数
    modified_schema = generate_executor_and_storage(
        execution_mode=request.execution_mode,
        remote_machine=request.remote_machine,
        tool_schema=modified_schema
    )

    logger.info("=" * 80)
    logger.info(f"[ModifyParams] 生成 Executor 和 Storage 后的Schema: {json.dumps(redact_secrets(modified_schema), ensure_ascii=False, indent=2)}")
    logger.info("=" * 80)

    # 提取修改后的参数
    modified_args = extract_arguments_from_schema(modified_schema)

    # 处理路径参数：确保所有路径都是绝对路径
    properties = modified_schema.get('input_schema', {}).get('properties', {})
    for param_name, param_info in properties.items():
        if param_name.endswith('_path') and param_name in modified_args:
            user_input = modified_args[param_name]
            if user_input and isinstance(user_input, str):
                # 确保路径是绝对路径
                if os.path.isabs(user_input):
                    # 已经是绝对路径，检查文件是否存在
                    if os.path.exists(user_input):
                        logger.info(f"[ModifyParams] 路径参数 {param_name}: 保持绝对路径 {user_input}")
                    else:
                        logger.warning(f"[ModifyParams] 路径参数 {param_name}: 绝对路径不存在 {user_input}")
                else:
                    # 相对路径，尝试在 session_files_dir 中查找
                    possible_path = os.path.join(session_files_dir, user_input)
                    if os.path.exists(possible_path):
                        modified_args[param_name] = possible_path
                        logger.info(f"[ModifyParams] 路径参数 {param_name}: 相对路径转绝对路径 {user_input} -> {possible_path}")
                    else:
                        # 尝试相对于当前工作目录
                        cwd_path = os.path.abspath(user_input)
                        if os.path.exists(cwd_path):
                            modified_args[param_name] = cwd_path
                            logger.info(f"[ModifyParams] 路径参数 {param_name}: 使用当前工作目录 {user_input} -> {cwd_path}")
                        else:
                            # 文件不存在，但仍然构建预期的绝对路径
                            modified_args[param_name] = possible_path
                            logger.warning(f"[ModifyParams] 路径参数 {param_name}: 文件不存在，使用预期路径 {user_input} -> {possible_path}")

    logger.info(f"[ModifyParams] 提取后的参数: {redact_secrets(modified_args)}")
    logger.info(f"[ModifyParams] Executor 参数: {redact_secrets(modified_args.get('executor'))}")
    logger.info(f"[ModifyParams] Storage 参数: {redact_secrets(modified_args.get('storage'))}")

    modified_args_store[session_id] = modified_args
    modified_schema_store[session_id] = modified_schema

    # 恢复agent执行
    pending_event = pending_events.get(session_id)
    if isinstance(pending_event, asyncio.Event):
        logger.info(f"[ModifyParams] 触发事件，恢复 agent 执行")
        pending_event.set()
    else:
        pending_events.pop(session_id, None)
        logger.warning(f"[ModifyParams] Session {session_id} 没有待处理的事件")

    return {"message": "参数已更新", "modified_args": modified_args}


@app.post("/api/terminate-execution")
async def terminate_execution(request: TerminateExecutionRequest):
    """终止正在执行的 agent 任务"""
    session_id = request.session_id

    logger.info("=" * 80)
    logger.info(f"[TerminateExecution] 收到终止执行请求")
    logger.info(f"[TerminateExecution] Session ID: {session_id}")
    logger.info("=" * 80)

    # 标记会话需要终止
    termination_requested[session_id] = True

    # 触发取消事件
    pending_event = pending_events.get(session_id)
    if isinstance(pending_event, asyncio.Event):
        # 设置终止事件，使 wait 立即返回
        cancel_event = cancel_execution_events.get(session_id)
        if not isinstance(cancel_event, asyncio.Event):
            cancel_event = asyncio.Event()
            cancel_execution_events[session_id] = cancel_event
        cancel_event.set()

        # 同时触发 pending_event 使其返回
        pending_event.set()

        logger.info(f"[TerminateExecution] 已触发会话 {session_id} 的终止信号")
        return {"message": "终止请求已发送", "status": "terminating"}
    else:
        pending_events.pop(session_id, None)
        cancel_execution_events.pop(session_id, None)
        logger.warning(f"[TerminateExecution] Session {session_id} 没有待处理的事件")
        return {"message": "没有正在执行的任务", "status": "no_active_task"}


@app.get("/api/files/{session_id}")
async def list_files(session_id: str):
    """获取会话文件列表"""
    session_dir = os.path.join(work_path, session_id, "files")
    logger.info(f"Listing files from: {session_dir}")
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
    logger.info(f"Found {len(files)} files")
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


@app.get("/api/download/{session_id}/{filename:path}")
async def download_file(session_id: str, filename: str):
    """下载文件 (支持子目录和可选的 files/ 前缀)"""
    # 兼容性处理：如果请求路径包含 files/ 前缀（例如前端根据文件系统路径拼接），则移除
    # 这样 /api/download/xxx/band.png 和 /api/download/xxx/files/band.png 都能工作
    clean_filename = filename
    if clean_filename.startswith("files/"):
        clean_filename = clean_filename[6:]
    elif clean_filename.startswith("/files/"):
        clean_filename = clean_filename[7:]
    
    file_path = os.path.join(work_path, session_id, "files", clean_filename)

    # 防止路径遍历攻击
    # ... (Normally we should check commonprefix, but assuming session_id isolation is enough for now for internal tool)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    return FileResponse(file_path, filename=os.path.basename(clean_filename))

@app.delete("/api/files/{session_id}/{filename:path}")
async def delete_file(session_id: str, filename: str):
    """删除文件"""
    # 同样的逻辑
    clean_filename = filename
    if clean_filename.startswith("files/"):
        clean_filename = clean_filename[6:]
        
    file_path = os.path.join(work_path, session_id, "files", clean_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        os.remove(file_path)
        return {"message": "文件已删除", "filename": clean_filename}
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
    logger.info(f"Loading sessions for {user_id} from {sessions_file}")
    
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
            
            logger.info(f"Loaded {len(sessions)} sessions")
            return {"sessions": sessions}
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {"sessions": []}
    logger.warning("Sessions file not found")
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
        logger.error(f"Error loading session history: {e}")
    
    return []


def update_session_history(user_id: str, session_id: str, history: List[List[str]], work_path: str):
    """更新用户会话列表中的历史记录"""
    logger.debug(f"Updating session history for User: {user_id}, Chat: {session_id}, History Len: {len(history)}")
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    
    if not os.path.exists(sessions_file):
        logger.warning(f"Sessions file not found: {sessions_file}")
        return

    try:
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        updated = False
        for session in sessions:
            # logger.debug(f"Checking session {session.get('chat_id')} against {session_id}")
            if session.get("chat_id") == session_id:
                session["history"] = history
                session["last_active"] = datetime.now().isoformat()
                session["message_count"] = len(history)
                updated = True
                logger.debug(f"Found and updated session {session_id}")
                break
        
        if updated:
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            logger.debug("Successfully saved sessions.json")
        else:
            logger.warning(f"Chat ID {session_id} not found in sessions.json")
            
    except Exception as e:
        logger.error(f"Failed to update session history in sessions.json: {e}")


@app.post("/api/user/{user_id}/sessions")
async def save_user_sessions(user_id: str, request: SaveSessionsRequest):
    """保存用户的所有聊天会话"""
    user_dir = os.path.join(work_path, user_id)
    os.makedirs(user_dir, exist_ok=True)
    sessions_file = os.path.join(user_dir, "sessions.json")
    logger.info(f"Saving {len(request.sessions)} sessions for {user_id} to {sessions_file}")
    
    try:
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(request.sessions, f, ensure_ascii=False, indent=2)
        return {"message": "Sessions saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save sessions: {str(e)}")


@app.get("/api/schema/{session_id}")
async def get_current_schema(session_id: str):
    """获取当前需要修改的参数schema"""
    schema = unmodified_schema_store.get(session_id, {})
    return {"schema": schema}


@app.get("/api/workflow-plan/{session_id}")
async def get_current_workflow_plan(session_id: str):
    """Return the workflow plan currently waiting for user review."""
    return {"workflow_plan": workflow_plan_store.get(session_id, {})}


@app.post("/api/workflow-plan/decision")
async def decide_workflow_plan(request: WorkflowPlanDecisionRequest):
    """Approve, reject, or run Pilot review for a pending workflow plan."""
    session_id = request.session_id
    decision = request.decision.lower().strip()
    if decision not in {"approved", "rejected", "pilot_review"}:
        raise HTTPException(status_code=400, detail="decision must be approved, rejected, or pilot_review")
    if session_id not in workflow_plan_store:
        raise HTTPException(status_code=404, detail="No workflow plan is waiting for review")

    if decision == "pilot_review":
        plan = workflow_plan_store[session_id]
        review = await generate_workflow_review(plan)
        plan["review"] = review
        workflow_plan_store[session_id] = plan
        await manager.send_message(session_id, {
            "type": "workflow_plan_required",
            "workflow_plan": plan,
        })
        await manager.send_message(session_id, {
            "type": "streaming_response",
            "content": render_workflow_plan_markdown(plan),
            "is_final": False,
        })
        return {
            "message": "Workflow plan reviewed by Pilot",
            "decision": decision,
            "workflow_plan": plan,
        }

    workflow_plan_decisions[session_id] = decision
    if session_id in workflow_plan_events:
        workflow_plan_events[session_id].set()
    if decision == "rejected":
        workflow_plan_store.pop(session_id, None)
        workflow_plan_tasks.pop(session_id, None)

    return {"message": f"Workflow plan {decision}", "decision": decision}


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "message": "Backend is running",
        "build_id": BACKEND_BUILD_ID,
        "capabilities": SERVER_CAPABILITIES,
    }

@app.get("/api/config")
async def get_config():
    """获取应用配置信息"""
    logger.info("收到配置请求")
    config = {
        "agent_info": agent_info,
        "mcp_server_url": mcp_server_url,
        "target_tools": target_tools,
        "build_id": BACKEND_BUILD_ID,
        "capabilities": SERVER_CAPABILITIES,
    }
    logger.debug(f"返回配置信息: {config}")
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

    registered_paths = {route.path for route in app.routes}
    logger.info(f"Backend build: {BACKEND_BUILD_ID}")
    logger.info(f"Backend module: {Path(__file__).resolve()}")
    logger.info(
        "Execution config route registered: "
        f"{'/api/execution-config' in registered_paths}"
    )

    # 加载MCP工具信息
    try:
        tools_info = asyncio.run(get_mcp_server_tools(mcp_server_url))
        logger.info(f"✅ 成功加载 {len(tools_info)} 个MCP工具")
    except Exception as e:
        logger.error(f"⚠️  加载MCP工具失败: {e}")
        tools_info = []

    # 配置静态文件服务 (如果在生产环境且存在dist)
    configure_static_serving()


def configure_static_serving():
    """配置前端静态文件服务"""
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # 尝试找到dist目录
    # app.py path: .../dptb-pilot/dptb_pilot/server/app.py
    # web_ui path: .../dptb-pilot/web_ui
    server_dir = os.path.dirname(__file__)
    pilot_pkg_dir = os.path.dirname(server_dir)
    project_root = os.path.dirname(pilot_pkg_dir)

    possible_paths = [
        os.path.join(project_root, "web_ui", "dist"), # Best for packaged/repo run
        os.path.join(os.getcwd(), "web_ui", "dist"),  # Best for local dev in root
    ]
    
    dist_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "index.html")):
            dist_path = path
            break
            
    if dist_path:
        logger.info(f"🎨 启用静态文件托管: {dist_path}")
        
        # 1. Mount assets
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
            
        # 2. Mount other static folders if needed (e.g. vite creates assets, maybe others?)
        # For safety, we can mount root, but it might shadow API.
        
        # 3. Catch-all route for SPA (Must be last)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # API和WebSocket已经被前面的路由捕获，这里只处理前端路由
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(status_code=404, detail="Not Found")
            
            # Check if file exists in dist (e.g. favicon.ico)
            file_path = os.path.join(dist_path, full_path)
            if os.path.isfile(file_path):
                 return FileResponse(file_path)
                 
            # 否则返回index.html (SPA路由)
            return FileResponse(os.path.join(dist_path, "index.html"))
            
        logger.info("✅ 前端静态服务已配置 (SPA Mode)")
    else:
        logger.info("ℹ️ 未发现前端编译产物，跳过静态服务配置 (请使用 npm run dev)")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
