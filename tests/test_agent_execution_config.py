import asyncio
from types import SimpleNamespace

import pytest

from dptb_pilot.core import agent


class FakeToolset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_mcp_toolset_overrides_model_executor_values():
    remote_executor = {
        "type": "dispatcher",
        "machine": {
            "batch_type": "Bohrium",
            "context_type": "Bohrium",
            "remote_profile": {},
        },
    }

    toolset = agent.mcp_tools(
        "http://localhost:50001/sse",
        {"executor": remote_executor},
    )

    assert toolset.override is True
    assert toolset.executor == {"type": "local"}
    assert toolset.storage == {"type": "local"}
    assert toolset.executor_map["run_abacus"] == remote_executor
    assert toolset.executor_map["run_abacus_scf_nscf"] == remote_executor
    assert toolset.executor_map["train_deeptb_model"] == remote_executor
    assert toolset.async_mode is False
    assert "band_predict_with_julia" in toolset.async_tool_names
    assert "convert_vasp_poscar_structure" not in agent.REMOTE_EXECUTION_TOOLS


def test_remote_storage_is_bound_to_calculation_toolset(monkeypatch):
    monkeypatch.setattr(agent, "CalculationMCPToolset", FakeToolset)
    storage = {
        "type": "bohrium",
        "username": "user@example.com",
        "password": "secret",
        "project_id": "123",
    }
    executor = {"type": "dispatcher", "machine": {"context_type": "Bohrium"}}

    toolset = agent.mcp_tools(
        "http://127.0.0.1:50002/sse",
        {"executor": executor, "storage": storage},
    )

    assert toolset.kwargs["storage"] == storage
    assert toolset.kwargs["executor_map"]["run_abacus_scf_nscf"] == executor
    assert toolset.kwargs["async_mode"] is True


def test_local_storage_remains_the_default(monkeypatch):
    monkeypatch.setattr(agent, "CalculationMCPToolset", FakeToolset)

    toolset = agent.mcp_tools("http://127.0.0.1:50002/sse")

    assert toolset.kwargs["storage"] == {"type": "local"}
    assert toolset.kwargs["async_mode"] is False


def test_mcp_timeouts_are_configurable(monkeypatch):
    monkeypatch.setattr(agent, "CalculationMCPToolset", FakeToolset)
    monkeypatch.setenv("MCP_TOOLS_CONNECT_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("MCP_TOOLS_SSE_READ_TIMEOUT_SECONDS", "7200")

    toolset = agent.mcp_tools("http://127.0.0.1:50002/sse")

    params = toolset.kwargs["connection_params"]
    assert params.timeout == 12
    assert params.sse_read_timeout == 7200


def test_only_remote_tools_use_async_submit_polling(monkeypatch):
    async def fake_get_tools(self, *args, **kwargs):
        async def run_async(*run_args, **run_kwargs):
            return None

        return [
            SimpleNamespace(name="band_predict_with_julia", run_async=run_async),
            SimpleNamespace(name="search_knowledge_base", run_async=run_async),
        ]

    monkeypatch.setattr(agent.CalculationMCPToolset, "get_tools", fake_get_tools)
    toolset = agent.BoundedCalculationMCPToolset(
        connection_params=agent.SseServerParams(url="http://127.0.0.1:50002/sse"),
        async_mode=False,
        remote_tool_names={"band_predict_with_julia"},
        async_tool_names={"band_predict_with_julia"},
    )

    tools = asyncio.run(toolset.get_tools())

    assert tools[0].async_mode is True
    assert tools[1].async_mode is False
    assert tools[1].executor == {"type": "local"}
    assert tools[1].storage == {"type": "local"}
    assert tools[1].wait is True


def test_local_tools_do_not_wait_for_remote_parallel_slots(monkeypatch):
    remote_started = asyncio.Event()
    release_remote = asyncio.Event()
    local_started = asyncio.Event()

    async def remote_run(*args, **kwargs):
        remote_started.set()
        await release_remote.wait()

    async def local_run(*args, **kwargs):
        local_started.set()

    async def fake_get_tools(self, *args, **kwargs):
        return [
            SimpleNamespace(name="run_abacus", run_async=remote_run),
            SimpleNamespace(name="read_file_content", run_async=local_run),
        ]

    monkeypatch.setattr(agent.CalculationMCPToolset, "get_tools", fake_get_tools)
    toolset = agent.BoundedCalculationMCPToolset(
        connection_params=agent.SseServerParams(url="http://127.0.0.1:50002/sse"),
        async_mode=False,
        max_parallelism=1,
        remote_tool_names={"run_abacus"},
        async_tool_names={"run_abacus"},
    )

    async def run_scenario():
        remote_tool, local_tool = await toolset.get_tools()
        remote_task = asyncio.create_task(remote_tool.run_async())
        await remote_started.wait()
        await asyncio.wait_for(local_tool.run_async(), timeout=0.5)
        assert local_started.is_set()
        release_remote.set()
        await remote_task

    asyncio.run(run_scenario())


def test_local_artifact_download_keeps_remote_storage(monkeypatch):
    remote_storage = {"type": "bohrium", "project_id": "123"}

    async def fake_get_tools(self, *args, **kwargs):
        async def run_async(*run_args, **run_kwargs):
            return None

        return [
            SimpleNamespace(
                name="download_artifact",
                run_async=run_async,
                storage=remote_storage,
            )
        ]

    monkeypatch.setattr(agent.CalculationMCPToolset, "get_tools", fake_get_tools)
    toolset = agent.BoundedCalculationMCPToolset(
        connection_params=agent.SseServerParams(url="http://127.0.0.1:50002/sse"),
        async_mode=False,
    )

    [tool] = asyncio.run(toolset.get_tools())

    assert tool.executor == {"type": "local"}
    assert tool.storage == remote_storage
    assert tool.async_mode is False


def test_remote_parallel_mode_waits_for_completed_results(monkeypatch):
    monkeypatch.setattr(agent, "CalculationMCPToolset", FakeToolset)
    executor = {"type": "dispatcher", "machine": {"context_type": "Bohrium"}}

    toolset = agent.mcp_tools(
        "http://127.0.0.1:50002/sse",
        {"executor": executor, "parallel_mode": "parallel", "max_parallelism": 4},
    )

    assert toolset.kwargs["wait"] is True


def test_remote_sequential_mode_keeps_waiting(monkeypatch):
    monkeypatch.setattr(agent, "CalculationMCPToolset", FakeToolset)
    executor = {"type": "dispatcher", "machine": {"context_type": "Bohrium"}}

    toolset = agent.mcp_tools(
        "http://127.0.0.1:50002/sse",
        {"executor": executor, "parallel_mode": "sequential"},
    )

    assert toolset.kwargs["wait"] is True


def test_build_execution_config_validates_parallel_limits():
    from dptb_pilot.server import app as server_app

    machine = {
        "type": "Bohrium",
        "config": {"username": "u", "password": "p", "project_id": "1"},
    }
    config = server_app.build_execution_config(
        "Remote", machine, parallel_mode="parallel", max_parallelism=99
    )
    assert config["parallel_mode"] == "parallel"
    assert config["max_parallelism"] == 32

    with pytest.raises(ValueError, match="parallel_mode"):
        server_app.build_execution_config("Remote", machine, parallel_mode="bad")
