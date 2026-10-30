import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import *
import os
import re
from functools import wraps
from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

_BASE_CALCULATION_MCP_TOOLSET = CalculationMCPToolset

from dptb_pilot.core.guardrail import tool_modify_guardrail


REMOTE_EXECUTION_TOOLS = {
    "band_with_baseline_model",
    "generate_sk_baseline_model",
    "run_lammps",
    "run_negf",
    "band_predict",
    "band_predict_with_julia",
    "dos_predict",
    "hamiltonian_predict",
    "hamiltonian_test",
    "run_abacus",
    "run_abacus_scf_nscf",
    "dpnegf_get_abacus_overlap",
    "dpnegf_run_lammps_task",
    "dpnegf_run_negf_task",
    "train_deepmd_model",
    "finetune_dpa3_model",
    "train_deeptb_model",
}

LOCAL_EXECUTOR = {"type": "local"}
LOCAL_STORAGE = {"type": "local"}
LOCAL_DATA_ACCESS_TOOLS = {
    "list_workspace_files",
    "read_file_content",
    "search_knowledge_base",
}


def _positive_float(value: object, default: float) -> float:
    """Return a positive float without letting bad runtime config break startup."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _valid_agent_identifier(value: object, fallback: str = "agent") -> str:
    """Normalize a value to the identifier format required by ADK agents."""
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))
    if not identifier:
        identifier = fallback
    if not (identifier[0].isalpha() or identifier[0] == "_"):
        identifier = f"_{identifier}"
    return identifier


def mcp_tools(mcp_tools_url, execution_config=None):
    execution_config = execution_config or {}
    remote_executor = execution_config.get("executor")
    storage = execution_config.get("storage") or LOCAL_STORAGE.copy()
    parallel_mode = str(execution_config.get("parallel_mode") or "sequential").lower()
    # ADK already executes multiple function calls from the same model turn
    # concurrently.  Waiting here controls result completeness, not whether
    # independent calls can overlap.  Returning a running job handle as a
    # completed tool response prevents the model from consuming the result and
    # continuing dependent work, so waiting is the safe default in every mode.
    configured_wait = execution_config.get("wait")
    if configured_wait is None:
        configured_wait = True
    try:
        max_parallelism = max(1, min(32, int(execution_config.get("max_parallelism", 1))))
    except (TypeError, ValueError):
        max_parallelism = 1
    connect_timeout = _positive_float(
        execution_config.get(
            "mcp_connect_timeout_seconds",
            os.getenv("MCP_TOOLS_CONNECT_TIMEOUT_SECONDS"),
        ),
        10.0,
    )
    sse_read_timeout = _positive_float(
        execution_config.get(
            "mcp_sse_read_timeout_seconds",
            os.getenv("MCP_TOOLS_SSE_READ_TIMEOUT_SECONDS"),
        ),
        3600.0,
    )
    configured_async_mode = execution_config.get("async_mode")
    if configured_async_mode is None:
        configured_async_mode = bool(remote_executor)
    executor_map = (
        {name: remote_executor for name in REMOTE_EXECUTION_TOOLS}
        if remote_executor
        else {}
    )
    toolset_class = (
        BoundedCalculationMCPToolset
        if CalculationMCPToolset is _BASE_CALCULATION_MCP_TOOLSET
        else CalculationMCPToolset
    )
    toolset_kwargs = dict(
        connection_params=SseServerParams(
            url=mcp_tools_url,
            timeout=connect_timeout,
            sse_read_timeout=sse_read_timeout,
        ),
        executor=LOCAL_EXECUTOR.copy(),
        storage=storage,
        executor_map=executor_map,
        async_mode=False,
        wait=bool(configured_wait),
        max_parallelism=max_parallelism,
        override=True,
    )
    if toolset_class is CalculationMCPToolset:
        toolset_kwargs.pop("max_parallelism", None)
        toolset_kwargs["async_mode"] = bool(configured_async_mode)
    else:
        toolset_kwargs["remote_tool_names"] = set(executor_map)
        toolset_kwargs["async_tool_names"] = (
            set(executor_map) if configured_async_mode else set()
        )
    return toolset_class(**toolset_kwargs)


class BoundedCalculationMCPToolset(CalculationMCPToolset):
    """Bound concurrency and use reconnectable polling for remote jobs."""

    def __init__(
        self,
        *args,
        max_parallelism: int = 1,
        remote_tool_names=None,
        async_tool_names=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_parallelism = max(1, min(32, int(max_parallelism)))
        self.remote_tool_names = set(remote_tool_names or ())
        self.async_tool_names = set(async_tool_names or ())
        self._remote_execution_semaphore = asyncio.Semaphore(self.max_parallelism)

    async def get_tools(self, *args, **kwargs):
        tools = await super().get_tools(*args, **kwargs)
        for tool in tools:
            is_remote = tool.name in self.remote_tool_names
            tool.async_mode = is_remote and tool.name in self.async_tool_names
            if not is_remote:
                # Remote execution is opt-in. Control-plane operations such as
                # workspace access, file reads and knowledge retrieval must not
                # enter remote submit/poll execution paths.
                tool.executor = LOCAL_EXECUTOR.copy()
                if tool.name in LOCAL_DATA_ACCESS_TOOLS:
                    tool.storage = LOCAL_STORAGE.copy()
                tool.wait = True
                tool.is_long_running = False
                continue

            original_run_async = tool.run_async

            @wraps(original_run_async)
            async def bounded_run_async(*run_args, _original=original_run_async, **run_kwargs):
                async with self._remote_execution_semaphore:
                    return await _original(*run_args, **run_kwargs)

            tool.run_async = bounded_run_async
        return tools


def create_llm_agent(
    session_id: str,
    mcp_tools_url: str,
    agent_info: dict,
    model_config: dict,
    execution_config: dict | None = None,
    agent_instance_id: str | None = None,
) -> LlmAgent:
    """Create an LLM Agent bound to a workspace and an optional chat instance."""

    instruction = agent_info['instruction']
    if "{session_id}" in instruction:
        # Use replace instead of format to avoid issues with other braces (e.g. JSON or LaTeX)
        instruction = instruction.replace("{session_id}", session_id)

    # DEBUG: Print instruction to verify prompt updates
    print("-" * 50)
    print("DEBUG: Current Agent System Instruction:")
    print(instruction)
    print("-" * 50)

    instance_id = agent_instance_id or session_id
    agent = LlmAgent(
        model=LiteLlm(**model_config),
        name=(
            f"{_valid_agent_identifier(agent_info['name'])}_"
            f"{_valid_agent_identifier(instance_id)}"
        ),
        description=agent_info['description'],
        instruction=instruction,
        tools=[mcp_tools(
            mcp_tools_url=mcp_tools_url,
            execution_config=execution_config,
        )],
        before_tool_callback=tool_modify_guardrail
    )

    return agent


