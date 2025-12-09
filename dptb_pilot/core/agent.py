from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm import *
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
import os
#from dp.agent.adapter.adk import CalculationMCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

from dptb_pilot.core.guardrail import tool_modify_guardrail


def mcp_tools(mcp_tools_url):
    return MCPToolset(
        connection_params=SseServerParams(url=mcp_tools_url)
    )


def create_llm_agent(session_id: str, mcp_tools_url: str, agent_info: dict, model_config: dict) -> LlmAgent:
    """根据用户信息创建LlmAgent"""

    instruction = agent_info['instruction']
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
        tools=[mcp_tools(mcp_tools_url=mcp_tools_url)],
        before_tool_callback=tool_modify_guardrail
    )

    return agent


