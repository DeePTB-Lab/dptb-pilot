import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from mcp.types import ResourceLink


PLUGIN_ROOT = Path(__file__).parents[1] / "codex_plugin" / "dptb-pilot"
SERVER_PATH = PLUGIN_ROOT / "mcp_server" / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("dptb_codex_plugin_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_server(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("DPTB_PILOT_CONFIG", str(config_path))
    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("PLUGIN_DATA", str(tmp_path / "plugin-data"))
    return _load_server()


def test_flat_config_migrates_to_v2(plugin_server, tmp_path):
    config_path = Path(os.environ["DPTB_PILOT_CONFIG"])
    config_path.write_text(json.dumps({
        "server_url": "localhost:50003/",
        "session_id": "s" * 32,
        "timeout_ms": 45000,
        "python_path": "python.exe",
    }), encoding="utf-8")

    config = plugin_server._load_config_data()

    assert config["version"] == 2
    assert config["connections"]["default"]["server_url"] == "http://localhost:50003"
    assert config["connections"]["default"]["session_id"] == "s" * 32
    assert config["strict_scientific_routing"] is True
    assert config["python_path"] == "python.exe"
    assert json.loads(config_path.read_text(encoding="utf-8"))["version"] == 2


def test_default_config_generates_32_character_session(plugin_server):
    config = plugin_server._load_config_data()
    session_id = config["connections"]["default"]["session_id"]
    assert len(session_id) == 32
    assert config["connections"]["default"]["server_url"] == "http://localhost:50003"


def test_bohrium_password_is_keyring_only_and_redacted(plugin_server, monkeypatch):
    stored = {}
    monkeypatch.setattr(plugin_server.keyring, "set_password", lambda service, user, value: stored.update({(service, user): value}))

    result = asyncio.run(plugin_server.dptb_upsert_execution_profile(
        "bohr-main",
        "bohrium",
        {"username": "user@example.com", "project_id": "42", "password": "must-not-persist"},
        bohrium_password="credential-secret",
    ))

    config_text = Path(os.environ["DPTB_PILOT_CONFIG"]).read_text(encoding="utf-8")
    assert "credential-secret" not in config_text
    assert "must-not-persist" not in config_text
    assert result["config"] == {"username": "user@example.com", "project_id": "42"}
    assert next(iter(stored.values())) == "credential-secret"


def test_file_resource_id_round_trips_nested_unicode_path(plugin_server):
    filename = "outputs/子目录/bands 01.json"
    assert plugin_server._decode_file_id(plugin_server._encode_file_id(filename)) == filename


def test_download_path_rejects_traversal_and_overwrite(plugin_server):
    with pytest.raises(RuntimeError, match="inside"):
        plugin_server._safe_output_path("../escape.dat", overwrite=False)

    output = plugin_server._safe_output_path("results/value.dat", overwrite=False)
    output.write_bytes(b"one")
    with pytest.raises(RuntimeError, match="overwrite"):
        plugin_server._safe_output_path("results/value.dat", overwrite=False)
    assert plugin_server._safe_output_path("results/value.dat", overwrite=True) == output


def test_artifact_listing_returns_clickable_resource_links(plugin_server, monkeypatch):
    async def fake_request(*_args, **_kwargs):
        return {"tasks": [{"id": "task", "artifacts": [{
            "id": "artifact-1",
            "file_name": "band.png",
            "mime_type": "image/png",
            "size": 123,
            "exists": True,
        }]}]}

    monkeypatch.setattr(plugin_server, "_request", fake_request)
    result = asyncio.run(plugin_server.dptb_list_artifacts())

    links = [item for item in result.content if isinstance(item, ResourceLink)]
    assert result.structuredContent["tasks"][0]["artifacts"][0]["id"] == "artifact-1"
    assert len(links) == 1
    assert str(links[0].uri) == "dptb://artifact/artifact-1"
    assert links[0].mimeType == "image/png"


def test_compatibility_ask_uses_submit_and_wait(plugin_server, monkeypatch):
    calls = []

    async def fake_submit(message, chat_id=None, workflow_mode=None):
        calls.append(("submit", message, chat_id, workflow_mode))
        return {"prompt": {"prompt_id": "prompt-1", "chat_id": "chat-1"}}

    async def fake_wait(identifier, chat_id, timeout_ms):
        calls.append(("wait", identifier, chat_id, timeout_ms))
        return {"run": {"status": "completed"}}

    monkeypatch.setattr(plugin_server, "dptb_submit_task", fake_submit)
    monkeypatch.setattr(plugin_server, "_wait_run", fake_wait)

    assert asyncio.run(plugin_server.dptb_ask("run it", timeout_ms=99))["run"]["status"] == "completed"
    assert calls == [("submit", "run it", None, None), ("wait", "prompt-1", "chat-1", 99)]


def test_strict_route_rejects_result_without_actual_tools(plugin_server, monkeypatch):
    async def fake_health():
        return {"capabilities": sorted(plugin_server.STRICT_CAPABILITIES)}

    async def fake_ask(*_args, **_kwargs):
        return {"run": {"status": "completed", "run_id": "r1", "provenance": {
            "engine": "dptb-pilot",
            "backend_build_id": "build",
            "tools": [],
        }}}

    monkeypatch.setattr(plugin_server, "_health", fake_health)
    monkeypatch.setattr(plugin_server, "_hook_status", lambda: {
        "config_present": True,
        "policy_script_present": True,
        "recent_scientific_activation": True,
    })
    monkeypatch.setattr(plugin_server, "dptb_ask", fake_ask)

    with pytest.raises(RuntimeError, match="lacks required Pilot provenance"):
        asyncio.run(plugin_server.dptb_route_scientific_task("calculate bands"))


def test_strict_hook_blocks_python_bypass_and_allows_dptb(tmp_path):
    policy = PLUGIN_ROOT / "hooks" / "strict_science_policy.py"
    env = {**os.environ, "PLUGIN_DATA": str(tmp_path)}
    prompt = subprocess.run(
        [sys.executable, str(policy), "user-prompt"],
        input=json.dumps({"session_id": "codex-session", "prompt": "calculate electronic structure"}),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert json.loads(prompt.stdout) == {}

    blocked = subprocess.run(
        [sys.executable, str(policy), "pre-tool"],
        input=json.dumps({"session_id": "codex-session", "tool_name": "exec_command", "tool_input": {"cmd": "python -c 'import numpy'"}}),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert json.loads(blocked.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"

    allowed = subprocess.run(
        [sys.executable, str(policy), "pre-tool"],
        input=json.dumps({"session_id": "codex-session", "tool_name": "mcp__dptb-pilot__dptb_route_scientific_task", "tool_input": {}}),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert json.loads(allowed.stdout) == {}
