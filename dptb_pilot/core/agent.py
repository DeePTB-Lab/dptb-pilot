from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import *
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
import os
from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

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
}


def mcp_tools(mcp_tools_url, execution_config=None):
    execution_config = execution_config or {}
    remote_executor = execution_config.get("executor")
    executor_map = (
        {name: remote_executor for name in REMOTE_EXECUTION_TOOLS}
        if remote_executor
        else {}
    )
    return CalculationMCPToolset(
        connection_params=SseServerParams(url=mcp_tools_url),
        executor={"type": "local"},
        storage={"type": "local"},
        executor_map=executor_map,
        override=True,
    )


def create_llm_agent(
    session_id: str,
    mcp_tools_url: str,
    agent_info: dict,
    model_config: dict,
    execution_config: dict | None = None,
) -> LlmAgent:
    """根据用户信息创建LlmAgent"""

    instruction = agent_info['instruction']
    if "{session_id}" in instruction:
        # Use replace instead of format to avoid issues with other braces (e.g. JSON or LaTeX)
        instruction = instruction.replace("{session_id}", session_id)

    # DEBUG: Print instruction to verify prompt updates
    print("-" * 50)
    print("DEBUG: Current Agent System Instruction:")
    print(instruction)
    print("-" * 50)

    agent = LlmAgent(
        model=LiteLlm(**model_config),
        name=f"{agent_info['name'].replace('-','_')}_{session_id}",
        description=agent_info['description'],
        instruction=instruction,
        tools=[mcp_tools(
            mcp_tools_url=mcp_tools_url,
            execution_config=execution_config,
        )],
        before_tool_callback=tool_modify_guardrail
    )

    return agent


