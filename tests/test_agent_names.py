import re

from dptb_pilot.core import agent


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_create_llm_agent_normalizes_session_key_for_adk_name(monkeypatch):
    monkeypatch.setattr(agent, "LlmAgent", FakeAgent)
    monkeypatch.setattr(agent, "LiteLlm", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent, "mcp_tools", lambda **kwargs: kwargs)

    created = agent.create_llm_agent(
        session_id="ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD:chat_1788154778534_bkev6eq",
        mcp_tools_url="http://localhost:50001/sse",
        agent_info={
            "name": "DeePTB-agent",
            "description": "test agent",
            "instruction": "Session: {session_id}",
        },
        model_config={"model": "test-model"},
    )

    assert created.kwargs["name"] == (
        "DeePTB_agent_ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD_"
        "chat_1788154778534_bkev6eq"
    )
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", created.kwargs["name"])
    assert created.kwargs["instruction"] == (
        "Session: ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD:chat_1788154778534_bkev6eq"
    )


def test_create_llm_agent_separates_workspace_session_from_chat_instance(monkeypatch):
    monkeypatch.setattr(agent, "LlmAgent", FakeAgent)
    monkeypatch.setattr(agent, "LiteLlm", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent, "mcp_tools", lambda **kwargs: kwargs)

    session_id = "ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD"
    chat_key = f"{session_id}:chat_1788154778534_bkev6eq"
    created = agent.create_llm_agent(
        session_id=session_id,
        agent_instance_id=chat_key,
        mcp_tools_url="http://localhost:50001/sse",
        agent_info={
            "name": "DeePTB-agent",
            "description": "test agent",
            "instruction": "Workspace: /personal/dptb-pilot/workspace/{session_id}/files",
        },
        model_config={"model": "test-model"},
    )

    assert created.kwargs["name"].endswith(
        "ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD_chat_1788154778534_bkev6eq"
    )
    assert created.kwargs["instruction"] == (
        "Workspace: /personal/dptb-pilot/workspace/"
        "ax6Fb6p04cymkUZ5KzachlTdaQDHYxCD/files"
    )
    assert ":chat_" not in created.kwargs["instruction"]


def test_valid_agent_identifier_handles_invalid_prefix_and_empty_value():
    assert agent._valid_agent_identifier("123/chat-id") == "_123_chat_id"
    assert agent._valid_agent_identifier("") == "agent"
