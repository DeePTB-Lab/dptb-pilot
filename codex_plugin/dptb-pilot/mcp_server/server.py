from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx
import keyring
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent


DEFAULT_SERVER_URL = "http://localhost:50003"
DEFAULT_TIMEOUT_MS = 30_000
MAX_INLINE_BYTES = 2 * 1024 * 1024
TERMINAL_RUN_STATES = {"completed", "failed", "cancelled", "interrupted"}
TERMINAL_GROUP_STATES = {"succeeded", "failed", "cancelled"}
STRICT_CAPABILITIES = {"rest_runtime_v1", "run_provenance"}
SECRET_KEYS = {"password", "authorization", "token", "secret", "api_key"}

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
mcp = FastMCP("dptb-pilot")

_config_lock = asyncio.Lock()
_login_lock = asyncio.Lock()
_logged_in: set[tuple[str, str]] = set()


def _config_candidates() -> list[Path]:
    explicit = os.getenv("DPTB_PILOT_CONFIG")
    if explicit:
        return [Path(explicit).expanduser()]
    return [Path("~/.dptb-pilot-codex/config.json").expanduser()]


def _config_path() -> Path:
    return _config_candidates()[0]


def _normalize_server_url(value: str | None) -> str:
    raw = str(value or DEFAULT_SERVER_URL).strip().rstrip("/")
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("DPTB Pilot server URL must be an http(s) URL.")
    return raw


def _validate_session_id(value: str | None, *, generate: bool = False) -> str:
    session_id = str(value or "").strip()
    if not session_id and generate:
        return secrets.token_hex(16)
    if len(session_id) != 32:
        raise RuntimeError("DPTB Pilot session_id must be exactly 32 characters.")
    return session_id


def _default_config() -> dict[str, Any]:
    return {
        "version": 2,
        "active_connection": "default",
        "connections": {
            "default": {
                "server_url": DEFAULT_SERVER_URL,
                "session_id": secrets.token_hex(16),
                "timeout_ms": DEFAULT_TIMEOUT_MS,
            }
        },
        "active_chat": None,
        "workflow_mode": "auto",
        "strict_scientific_routing": True,
        "execution": {
            "mode": "Local",
            "profile_id": None,
            "parallel_mode": "auto",
            "max_parallelism": 4,
        },
        "execution_profiles": {},
        "download_directory": ".dptb-pilot-downloads",
        "matterviz_uri": "matterviz://open?path={path}",
    }


def _migrate_config(raw: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw, dict):
        return _default_config(), True
    if int(raw.get("version") or 1) >= 2 and isinstance(raw.get("connections"), dict):
        data = _default_config()
        data.update(raw)
        data["version"] = 2
        data["connections"] = dict(raw.get("connections") or data["connections"])
        data["execution_profiles"] = dict(raw.get("execution_profiles") or {})
        data["execution"] = {**_default_config()["execution"], **dict(raw.get("execution") or {})}
        return data, data != raw

    data = _default_config()
    data["connections"]["default"] = {
        "server_url": _normalize_server_url(raw.get("server_url")),
        "session_id": _validate_session_id(raw.get("session_id"), generate=True),
        "timeout_ms": int(raw.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
    }
    for key in ("python_path", "download_directory", "matterviz_uri"):
        if raw.get(key):
            data[key] = raw[key]
    if raw.get("chat_id"):
        data["active_chat"] = raw["chat_id"]
    return data, True


def _write_config(data: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_config_data() -> dict[str, Any]:
    path = next((candidate for candidate in _config_candidates() if candidate.exists()), None)
    raw: Any = {}
    if path:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read DPTB Pilot config: {exc}") from exc
    data, changed = _migrate_config(raw)
    if path is None or changed:
        _write_config(data)
    return data


def _save_config(data: dict[str, Any]) -> None:
    data["version"] = 2
    _write_config(data)


def _active_connection(data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or _load_config_data()
    connections = data.get("connections") or {}
    profile_id = str(data.get("active_connection") or "default")
    connection = connections.get(profile_id)
    if not isinstance(connection, dict):
        raise RuntimeError(f"Active DPTB Pilot connection profile does not exist: {profile_id}")
    server_url = os.getenv("DPTB_PILOT_SERVER_URL") or connection.get("server_url")
    session_id = os.getenv("DPTB_PILOT_SESSION_ID") or connection.get("session_id")
    timeout_ms = os.getenv("DPTB_PILOT_TIMEOUT_MS") or connection.get("timeout_ms")
    return {
        "profile_id": profile_id,
        "server_url": _normalize_server_url(server_url),
        "session_id": _validate_session_id(session_id),
        "timeout_ms": max(1_000, int(timeout_ms or DEFAULT_TIMEOUT_MS)),
    }


def _active_chat(data: dict[str, Any] | None = None, chat_id: str | None = None) -> str:
    data = data or _load_config_data()
    connection = _active_connection(data)
    return str(chat_id or data.get("active_chat") or connection["session_id"])


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if key.lower() in SECRET_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _url(connection: dict[str, Any], path: str) -> str:
    return f"{connection['server_url']}/{path.lstrip('/')}"


async def _ensure_login(connection: dict[str, Any]) -> None:
    key = (connection["server_url"], connection["session_id"])
    if key in _logged_in:
        return
    async with _login_lock:
        if key in _logged_in:
            return
        timeout = httpx.Timeout(connection["timeout_ms"] / 1000)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _url(connection, "/api/login"),
                json={"session_id": connection["session_id"]},
            )
            _raise_for_status(response)
        _logged_in.add(key)


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = response.text[:2_000]
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload.get("message") or payload)
    except (ValueError, json.JSONDecodeError):
        pass
    raise RuntimeError(f"DPTB Pilot API error ({response.status_code}): {detail}")


async def _request(
    method: str,
    path: str,
    *,
    json_body: Any | None = None,
    files: Any | None = None,
    login: bool = True,
) -> Any:
    connection = _active_connection()
    if login:
        await _ensure_login(connection)
    timeout = httpx.Timeout(connection["timeout_ms"] / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            _url(connection, path),
            json=json_body,
            files=files,
        )
    _raise_for_status(response)
    if not response.content:
        return None
    return response.json()


async def _download_bytes(remote_path: str) -> tuple[bytes, str]:
    connection = _active_connection()
    await _ensure_login(connection)
    timeout = httpx.Timeout(connection["timeout_ms"] / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(_url(connection, remote_path))
    _raise_for_status(response)
    return response.content, response.headers.get("content-type", "application/octet-stream").split(";", 1)[0]


def _encode_file_id(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_file_id(value: str) -> str:
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(value + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("Invalid DPTB file resource id.") from exc
    if not decoded or "\x00" in decoded:
        raise RuntimeError("Invalid DPTB file resource id.")
    return decoded.replace("\\", "/").lstrip("/")


def _file_resource_link(file_info: dict[str, Any]) -> ResourceLink:
    name = str(file_info.get("name") or "file")
    mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return ResourceLink(
        type="resource_link",
        name=name,
        uri=f"dptb://file/{_encode_file_id(name)}",
        mimeType=mime_type,
        size=int(file_info.get("size") or 0) or None,
        description="DPTB Pilot session file",
    )


def _artifact_resource_link(artifact: dict[str, Any]) -> ResourceLink:
    name = str(artifact.get("file_name") or artifact.get("workspace_path") or artifact.get("source_path") or artifact.get("id"))
    return ResourceLink(
        type="resource_link",
        name=Path(name.replace("\\", "/")).name or name,
        uri=f"dptb://artifact/{artifact['id']}",
        mimeType=str(artifact.get("mime_type") or "application/octet-stream"),
        size=int(artifact.get("size") or 0) or None,
        description="DPTB Pilot task artifact",
    )


def _call_result(data: dict[str, Any], links: list[ResourceLink] | None = None) -> CallToolResult:
    content: list[Any] = [TextContent(type="text", text=json.dumps(_redact(data), ensure_ascii=False, indent=2, default=str))]
    content.extend(links or [])
    return CallToolResult(content=content, structuredContent=_redact(data))


def _preview_result(name: str, payload: bytes, mime_type: str, metadata: dict[str, Any]) -> CallToolResult:
    content: list[Any] = []
    if mime_type.startswith("image/") and len(payload) <= MAX_INLINE_BYTES:
        content.append(ImageContent(type="image", data=base64.b64encode(payload).decode("ascii"), mimeType=mime_type))
    elif (mime_type.startswith("text/") or Path(name).suffix.lower() in {".csv", ".json", ".yaml", ".yml", ".md", ".log", ".out", ".xml"}) and len(payload) <= MAX_INLINE_BYTES:
        content.append(TextContent(type="text", text=payload.decode("utf-8", errors="replace")))
    else:
        content.append(TextContent(type="text", text=json.dumps(metadata, ensure_ascii=False, indent=2)))
    return CallToolResult(content=content, structuredContent=metadata)


def _download_root(data: dict[str, Any] | None = None) -> Path:
    data = data or _load_config_data()
    configured = Path(str(data.get("download_directory") or ".dptb-pilot-downloads")).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    workspace = Path(os.getenv("CODEX_WORKSPACE_ROOT") or Path.cwd())
    return (workspace / configured).resolve()


def _safe_output_path(output_name: str, *, overwrite: bool) -> Path:
    root = _download_root()
    root.mkdir(parents=True, exist_ok=True)
    candidate = (root / output_name.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Download path must remain inside the configured download directory.") from exc
    if candidate.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing file: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _keyring_service() -> str:
    return f"dptb-pilot-codex:{_config_path().resolve()}"


def _password_env_name(profile_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]", "_", profile_id).upper()
    return f"DPTB_BOHRIUM_PASSWORD_{suffix}"


def _get_bohrium_password(profile_id: str) -> str:
    password = os.getenv(_password_env_name(profile_id)) or os.getenv("DPTB_BOHRIUM_PASSWORD")
    if not password:
        password = keyring.get_password(_keyring_service(), profile_id)
    if not password:
        raise RuntimeError(f"No Bohrium password is stored for execution profile '{profile_id}'.")
    return password


def _hook_data_root() -> Path:
    configured = os.getenv("PLUGIN_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path("~/.dptb-pilot-codex/hook-data").expanduser().resolve()


def _hook_status(max_age_seconds: int = 900) -> dict[str, Any]:
    hook_config = PLUGIN_ROOT / "hooks" / "hooks.json"
    policy_script = PLUGIN_ROOT / "hooks" / "strict_science_policy.py"
    directory = _hook_data_root() / "strict-science"
    newest: dict[str, Any] | None = None
    newest_mtime = 0.0
    if directory.is_dir():
        for path in directory.glob("*.json"):
            try:
                mtime = path.stat().st_mtime
                if mtime > newest_mtime:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, dict):
                        newest = value
                        newest_mtime = mtime
            except (OSError, json.JSONDecodeError):
                continue
    age = max(0.0, time.time() - newest_mtime) if newest_mtime else None
    return {
        "config_present": hook_config.is_file(),
        "policy_script_present": policy_script.is_file(),
        "observed": newest is not None,
        "recent_scientific_activation": bool(newest and age is not None and age <= max_age_seconds and newest.get("scientific")),
        "last_event_age_seconds": round(age, 3) if age is not None else None,
        "last_event": _redact(newest) if newest else None,
    }


async def _health() -> dict[str, Any]:
    value = await _request("GET", "/api/health", login=False)
    return value if isinstance(value, dict) else {}


async def _wait_run(identifier: str, chat_id: str, timeout_ms: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, timeout_ms) / 1000
    latest: dict[str, Any] = {}
    while True:
        remaining = deadline - time.monotonic()
        wait_seconds = max(0.0, min(30.0, remaining))
        connection = _active_connection()
        latest = await _request(
            "GET",
            f"/api/runtime/{quote(connection['session_id'])}/{quote(chat_id)}/runs/{quote(identifier)}?wait_seconds={wait_seconds}",
        )
        run = latest.get("run") if isinstance(latest, dict) else None
        if isinstance(run, dict) and str(run.get("status")) in TERMINAL_RUN_STATES:
            return latest
        if remaining <= 0:
            return latest


async def _wait_group(group_id: str, timeout_ms: int) -> dict[str, Any]:
    deadline = time.monotonic() + max(0, timeout_ms) / 1000
    latest: dict[str, Any] = {}
    while True:
        latest = await _request("GET", f"/api/task-groups/{quote(group_id)}")
        if str(latest.get("status")) in TERMINAL_GROUP_STATES:
            return latest
        if time.monotonic() >= deadline:
            return latest
        await asyncio.sleep(0.5)


async def _find_artifact(artifact_id: str) -> dict[str, Any]:
    listing = await dptb_list_artifacts()
    data = listing.structuredContent or {}
    for task in data.get("tasks", []):
        for artifact in task.get("artifacts", []):
            if str(artifact.get("id")) == artifact_id:
                return artifact
    raise RuntimeError(f"DPTB Pilot artifact does not exist: {artifact_id}")


@mcp.tool()
async def dptb_configure_connection(
    server_url: str = DEFAULT_SERVER_URL,
    session_id: str | None = None,
    profile_id: str = "default",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    activate: bool = True,
) -> dict[str, Any]:
    """Create or update a DPTB Pilot connection profile. A 32-character session id is generated when omitted."""
    async with _config_lock:
        data = _load_config_data()
        normalized = {
            "server_url": _normalize_server_url(server_url),
            "session_id": _validate_session_id(session_id, generate=True),
            "timeout_ms": max(1_000, int(timeout_ms)),
        }
        data.setdefault("connections", {})[profile_id] = normalized
        if activate:
            data["active_connection"] = profile_id
            data["active_chat"] = None
        _save_config(data)
        _logged_in.clear()
    return {"profile_id": profile_id, "active": activate, **normalized}


@mcp.tool()
async def dptb_status() -> dict[str, Any]:
    """Check connection, backend capabilities, active chat, and strict scientific routing readiness."""
    data = _load_config_data()
    connection = _active_connection(data)
    health = await _health()
    await _ensure_login(connection)
    app_config = await _request("GET", "/api/config")
    capabilities = set(health.get("capabilities") or app_config.get("capabilities") or [])
    hooks = _hook_status()
    return {
        "server_url": connection["server_url"],
        "session_id_prefix": connection["session_id"][:4],
        "active_chat": data.get("active_chat"),
        "health": health,
        "config": _redact(app_config),
        "strict_scientific_routing": {
            "enabled": True,
            "required_capabilities": sorted(STRICT_CAPABILITIES),
            "backend_ready": STRICT_CAPABILITIES.issubset(capabilities),
            "hooks": hooks,
            "ready_for_current_scientific_turn": STRICT_CAPABILITIES.issubset(capabilities) and hooks["recent_scientific_activation"],
        },
    }


@mcp.tool()
async def dptb_list_chats() -> dict[str, Any]:
    """List DPTB Pilot chats for the active session."""
    connection = _active_connection()
    return await _request("GET", f"/api/user/{quote(connection['session_id'])}/sessions")


@mcp.tool()
async def dptb_create_chat(title: str | None = None, chat_id: str | None = None, select: bool = True) -> dict[str, Any]:
    """Create a DPTB Pilot chat and optionally make it active."""
    connection = _active_connection()
    result = await _request(
        "POST",
        f"/api/user/{quote(connection['session_id'])}/sessions/new",
        json_body={"chat_id": chat_id, "title": title},
    )
    created_id = str(result["session"]["chat_id"])
    if select:
        async with _config_lock:
            data = _load_config_data()
            data["active_chat"] = created_id
            _save_config(data)
    return result


@mcp.tool()
async def dptb_select_chat(chat_id: str) -> dict[str, Any]:
    """Select an existing DPTB Pilot chat as the default for subsequent tools."""
    chats = await dptb_list_chats()
    if chat_id not in {str(item.get("chat_id")) for item in chats.get("sessions", [])}:
        raise RuntimeError(f"DPTB Pilot chat does not exist: {chat_id}")
    async with _config_lock:
        data = _load_config_data()
        data["active_chat"] = chat_id
        _save_config(data)
    return {"active_chat": chat_id}


@mcp.tool()
async def dptb_delete_chat(chat_id: str) -> dict[str, Any]:
    """Delete a DPTB Pilot chat and its isolated runtime state."""
    connection = _active_connection()
    result = await _request("DELETE", f"/api/user/{quote(connection['session_id'])}/sessions/{quote(chat_id)}")
    async with _config_lock:
        data = _load_config_data()
        if data.get("active_chat") == chat_id:
            data["active_chat"] = None
            _save_config(data)
    return result


@mcp.tool()
async def dptb_get_history(chat_id: str | None = None) -> dict[str, Any]:
    """Get persisted history for a DPTB Pilot chat."""
    connection = _active_connection()
    resolved = _active_chat(chat_id=chat_id)
    return await _request("GET", f"/api/user/{quote(connection['session_id'])}/sessions/{quote(resolved)}/history")


@mcp.tool()
async def dptb_submit_task(message: str, chat_id: str | None = None, workflow_mode: str | None = None) -> dict[str, Any]:
    """Submit one asynchronous task to the DPTB Pilot FIFO runtime."""
    data = _load_config_data()
    connection = _active_connection(data)
    resolved = _active_chat(data, chat_id)
    return await _request(
        "POST",
        "/api/runtime/enqueue",
        json_body={
            "session_id": connection["session_id"],
            "chat_id": resolved,
            "message": message,
            "workflow_mode": workflow_mode or data.get("workflow_mode") or "auto",
            "protocol_version": 2,
        },
    )


@mcp.tool()
async def dptb_submit_tasks(
    tasks: list[dict[str, Any]],
    base_chat_id: str | None = None,
    parallel_mode: str | None = None,
    max_parallelism: int | None = None,
    failure_policy: str = "continue_independent",
) -> dict[str, Any]:
    """Submit independent or depends_on-linked DPTB Pilot tasks as a bounded task graph."""
    data = _load_config_data()
    connection = _active_connection(data)
    execution = data.get("execution") or {}
    return await _request(
        "POST",
        "/api/runtime/batch",
        json_body={
            "session_id": connection["session_id"],
            "base_chat_id": base_chat_id,
            "tasks": tasks,
            "parallel_mode": parallel_mode or execution.get("parallel_mode") or "auto",
            "max_parallelism": max_parallelism or execution.get("max_parallelism") or 4,
            "failure_policy": failure_policy,
        },
    )


@mcp.tool()
async def dptb_get_runtime(chat_id: str | None = None) -> dict[str, Any]:
    """Get active run, queued prompts, and recent runs for one DPTB Pilot chat."""
    connection = _active_connection()
    resolved = _active_chat(chat_id=chat_id)
    return await _request("GET", f"/api/runtime/{quote(connection['session_id'])}/{quote(resolved)}")


@mcp.tool()
async def dptb_wait_run(identifier: str, chat_id: str | None = None, timeout_ms: int = 120_000) -> dict[str, Any]:
    """Wait for a DPTB Pilot prompt/run id, returning stages, tools, usage, charges, and provenance."""
    return await _wait_run(identifier, _active_chat(chat_id=chat_id), timeout_ms)


@mcp.tool()
async def dptb_remove_queued_task(prompt_id: str, chat_id: str | None = None) -> dict[str, Any]:
    """Remove a task that is still queued; an active run is never removed."""
    connection = _active_connection()
    resolved = _active_chat(chat_id=chat_id)
    return await _request("DELETE", f"/api/runtime/{quote(connection['session_id'])}/{quote(resolved)}/queue/{quote(prompt_id)}")


@mcp.tool()
async def dptb_get_task_group(group_id: str) -> dict[str, Any]:
    """Get a parallel DPTB Pilot task-group snapshot."""
    return await _request("GET", f"/api/task-groups/{quote(group_id)}")


@mcp.tool()
async def dptb_wait_task_group(group_id: str, timeout_ms: int = 120_000) -> dict[str, Any]:
    """Wait for a DPTB Pilot task group to finish or reach the timeout."""
    return await _wait_group(group_id, timeout_ms)


@mcp.tool()
async def dptb_cancel_task_group(group_id: str) -> dict[str, Any]:
    """Cancel queued/running nodes in a DPTB Pilot task group."""
    return await _request("POST", f"/api/task-groups/{quote(group_id)}/cancel")


@mcp.tool()
async def dptb_ask(message: str, chat_id: str | None = None, timeout_ms: int = 300_000) -> dict[str, Any]:
    """Compatibility entry point implemented as submit plus wait on the shared runtime."""
    submitted = await dptb_submit_task(message, chat_id=chat_id)
    prompt = submitted.get("prompt") or {}
    resolved = str(prompt.get("chat_id") or _active_chat(chat_id=chat_id))
    result = await _wait_run(str(prompt["prompt_id"]), resolved, timeout_ms)
    run = result.get("run") or {}
    if str(run.get("status")) not in TERMINAL_RUN_STATES:
        raise RuntimeError("DPTB Pilot task did not finish before the compatibility timeout.")
    return result


@mcp.tool()
async def dptb_get_workflow_plan(chat_id: str | None = None) -> dict[str, Any]:
    """Get the DPTB Pilot workflow plan waiting for review."""
    connection = _active_connection()
    resolved = _active_chat(chat_id=chat_id)
    return await _request("GET", f"/api/workflow-plan/{quote(connection['session_id'])}?chat_id={quote(resolved)}")


@mcp.tool()
async def dptb_decide_workflow(decision: str, chat_id: str | None = None) -> dict[str, Any]:
    """Approve, reject, or request pilot_review for a pending DPTB workflow."""
    connection = _active_connection()
    return await _request(
        "POST",
        "/api/workflow-plan/decision",
        json_body={"session_id": connection["session_id"], "chat_id": _active_chat(chat_id=chat_id), "decision": decision},
    )


@mcp.tool()
async def dptb_get_pending_schema(chat_id: str | None = None) -> dict[str, Any]:
    """Get the current DPTB tool-parameter schema waiting for confirmation."""
    connection = _active_connection()
    resolved = _active_chat(chat_id=chat_id)
    return await _request("GET", f"/api/schema/{quote(connection['session_id'])}?chat_id={quote(resolved)}")


@mcp.tool()
async def dptb_submit_parameters(modified_schema: dict[str, Any], chat_id: str | None = None) -> dict[str, Any]:
    """Submit structured parameter values for a pending DPTB tool call."""
    data = _load_config_data()
    connection = _active_connection(data)
    execution = data.get("execution") or {}
    remote_machine = _resolved_remote_machine(data, execution.get("profile_id")) if execution.get("mode") == "Remote" else None
    return await _request(
        "POST",
        "/api/modify-params",
        json_body={
            "session_id": connection["session_id"],
            "chat_id": _active_chat(data, chat_id),
            "modified_schema": modified_schema,
            "execution_mode": execution.get("mode") or "Local",
            "selected_machine_id": execution.get("profile_id"),
            "remote_machine": remote_machine,
        },
    )


@mcp.tool()
async def dptb_list_execution_profiles() -> dict[str, Any]:
    """List local/Bohrium/Slurm profiles without returning stored secrets."""
    data = _load_config_data()
    return {"selected": data.get("execution"), "profiles": _redact(data.get("execution_profiles") or {})}


@mcp.tool()
async def dptb_upsert_execution_profile(
    profile_id: str,
    profile_type: str,
    config: dict[str, Any],
    display_name: str | None = None,
    bohrium_password: str | None = None,
) -> dict[str, Any]:
    """Create/update a Bohrium or Slurm profile. Bohrium passwords go only to the OS credential store."""
    normalized_type = profile_type.strip().lower()
    if normalized_type not in {"bohrium", "slurm"}:
        raise RuntimeError("profile_type must be bohrium or slurm.")
    safe_config = {key: value for key, value in config.items() if key.lower() not in SECRET_KEYS}
    if normalized_type == "bohrium":
        missing = [key for key in ("username", "project_id") if not safe_config.get(key)]
        if missing:
            raise RuntimeError(f"Bohrium profile is missing: {', '.join(missing)}")
        if bohrium_password:
            keyring.set_password(_keyring_service(), profile_id, bohrium_password)
    else:
        missing = [key for key in ("hostname", "username", "remote_root", "key_filename") if not safe_config.get(key)]
        if missing:
            raise RuntimeError(f"Slurm profile is missing: {', '.join(missing)}")
    async with _config_lock:
        data = _load_config_data()
        record = {"id": profile_id, "name": display_name or profile_id, "type": normalized_type, "config": safe_config}
        data.setdefault("execution_profiles", {})[profile_id] = record
        _save_config(data)
    return _redact(record)


@mcp.tool()
async def dptb_select_execution_profile(profile_id: str | None = None, execution_mode: str = "Remote") -> dict[str, Any]:
    """Select the local mode or a saved remote execution profile."""
    mode = execution_mode.title()
    if mode not in {"Local", "Remote"}:
        raise RuntimeError("execution_mode must be Local or Remote.")
    async with _config_lock:
        data = _load_config_data()
        if mode == "Remote" and profile_id not in (data.get("execution_profiles") or {}):
            raise RuntimeError(f"Execution profile does not exist: {profile_id}")
        data["execution"] = {**data.get("execution", {}), "mode": mode, "profile_id": profile_id if mode == "Remote" else None}
        _save_config(data)
    return _redact(data["execution"])


def _resolved_remote_machine(data: dict[str, Any], profile_id: str | None) -> dict[str, Any]:
    profile = (data.get("execution_profiles") or {}).get(profile_id or "")
    if not isinstance(profile, dict):
        raise RuntimeError(f"Execution profile does not exist: {profile_id}")
    machine = {"type": profile.get("type"), "config": dict(profile.get("config") or {})}
    if str(profile.get("type")).lower() == "bohrium":
        machine["config"]["password"] = _get_bohrium_password(str(profile_id))
    return machine


@mcp.tool()
async def dptb_apply_execution_config(
    execution_mode: str | None = None,
    profile_id: str | None = None,
    parallel_mode: str | None = None,
    max_parallelism: int | None = None,
) -> dict[str, Any]:
    """Apply Local/Bohrium/Slurm execution settings to the active DPTB Pilot session."""
    async with _config_lock:
        data = _load_config_data()
        execution = {**data.get("execution", {})}
        if execution_mode:
            execution["mode"] = execution_mode.title()
        if profile_id is not None:
            execution["profile_id"] = profile_id
        if parallel_mode:
            execution["parallel_mode"] = parallel_mode.lower()
        if max_parallelism is not None:
            execution["max_parallelism"] = max(1, min(32, int(max_parallelism)))
        if execution.get("mode") not in {"Local", "Remote"}:
            raise RuntimeError("execution_mode must be Local or Remote.")
        if execution.get("parallel_mode") not in {"sequential", "auto", "parallel"}:
            raise RuntimeError("parallel_mode must be sequential, auto, or parallel.")
        remote_machine = _resolved_remote_machine(data, execution.get("profile_id")) if execution.get("mode") == "Remote" else None
        connection = _active_connection(data)
        result = await _request(
            "POST",
            "/api/execution-config",
            json_body={
                "session_id": connection["session_id"],
                "execution_mode": execution.get("mode"),
                "selected_machine_id": execution.get("profile_id"),
                "remote_machine": remote_machine,
                "parallel_mode": execution.get("parallel_mode"),
                "max_parallelism": execution.get("max_parallelism"),
            },
        )
        data["execution"] = execution
        _save_config(data)
    return _redact(result)


@mcp.tool()
async def dptb_list_files() -> CallToolResult:
    """List session files as structured data and clickable dptb:// resources."""
    connection = _active_connection()
    data = await _request("GET", f"/api/files/{quote(connection['session_id'])}")
    return _call_result(data, [_file_resource_link(item) for item in data.get("files", [])])


@mcp.tool()
async def dptb_upload_file(local_path: str, remote_name: str | None = None) -> dict[str, Any]:
    """Upload one local path to the DPTB Pilot session workspace."""
    path = Path(local_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"File does not exist: {path}")
    connection = _active_connection()
    with path.open("rb") as handle:
        return await _request(
            "POST",
            f"/api/upload/{quote(connection['session_id'])}",
            files=[("files", (remote_name or path.name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream"))],
        )


@mcp.tool()
async def dptb_upload_files(local_paths: list[str]) -> dict[str, Any]:
    """Upload multiple local files in one DPTB Pilot request."""
    paths = [Path(value).expanduser().resolve() for value in local_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Files do not exist: {missing}")
    handles = [path.open("rb") for path in paths]
    try:
        files = [
            ("files", (path.name, handle, mimetypes.guess_type(path.name)[0] or "application/octet-stream"))
            for path, handle in zip(paths, handles)
        ]
        connection = _active_connection()
        return await _request("POST", f"/api/upload/{quote(connection['session_id'])}", files=files)
    finally:
        for handle in handles:
            handle.close()


@mcp.tool()
async def dptb_preview_file(filename: str) -> CallToolResult:
    """Preview an image/text file natively; large or binary files return metadata only."""
    connection = _active_connection()
    listing = await _request("GET", f"/api/files/{quote(connection['session_id'])}")
    info = next((item for item in listing.get("files", []) if str(item.get("name")) == filename), None)
    if info is None:
        raise RuntimeError(f"DPTB Pilot file does not exist: {filename}")
    metadata = {**info, "resource_uri": f"dptb://file/{_encode_file_id(filename)}"}
    if int(info.get("size") or 0) > MAX_INLINE_BYTES:
        return _preview_result(filename, b"", "application/octet-stream", metadata)
    payload, content_type = await _download_bytes(f"/api/download/{quote(connection['session_id'])}/{quote(filename, safe='/')}")
    return _preview_result(filename, payload, content_type, metadata)


@mcp.tool()
async def dptb_delete_file(filename: str) -> dict[str, Any]:
    """Delete one file from the DPTB Pilot session workspace."""
    connection = _active_connection()
    return await _request("DELETE", f"/api/files/{quote(connection['session_id'])}/{quote(filename, safe='/')}")


@mcp.tool()
async def dptb_download_file(filename: str, output_name: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Download a session file into the configured workspace-controlled download directory."""
    connection = _active_connection()
    payload, _ = await _download_bytes(f"/api/download/{quote(connection['session_id'])}/{quote(filename, safe='/')}")
    target = _safe_output_path(output_name or Path(filename).name, overwrite=overwrite)
    target.write_bytes(payload)
    return {"filename": filename, "output_path": str(target), "bytes": len(payload)}


@mcp.tool()
async def dptb_list_artifacts() -> CallToolResult:
    """List task artifacts as structured data and clickable dptb://artifact resources."""
    connection = _active_connection()
    data = await _request("GET", f"/api/artifacts/{quote(connection['session_id'])}")
    links = [_artifact_resource_link(item) for task in data.get("tasks", []) for item in task.get("artifacts", [])]
    return _call_result(data, links)


@mcp.tool()
async def dptb_stage_artifact(artifact_id: str) -> CallToolResult:
    """Stage a recorded remote artifact into the DPTB Pilot session workspace."""
    connection = _active_connection()
    data = await _request("POST", f"/api/artifacts/{quote(connection['session_id'])}/{quote(artifact_id)}/stage")
    artifact = data.get("artifact") or {}
    return _call_result(data, [_artifact_resource_link(artifact)] if artifact.get("id") else [])


@mcp.tool()
async def dptb_download_artifact(artifact_id: str, output_name: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Stage if needed, then download a selected DPTB artifact locally."""
    artifact = await _find_artifact(artifact_id)
    if not artifact.get("exists"):
        staged = await dptb_stage_artifact(artifact_id)
        artifact = (staged.structuredContent or {}).get("artifact") or artifact
    workspace_path = str(artifact.get("workspace_path") or "")
    if not workspace_path:
        raise RuntimeError("Artifact is not stageable and has no workspace file.")
    return await dptb_download_file(workspace_path, output_name or artifact.get("file_name"), overwrite)


@mcp.tool()
async def dptb_get_usage(chat_id: str | None = None) -> dict[str, Any]:
    """Get DPTB Pilot input/output/cached/reasoning/total tokens and billing aggregates."""
    connection = _active_connection()
    suffix = f"?chat_id={quote(chat_id)}" if chat_id else ""
    return await _request("GET", f"/api/usage/{quote(connection['session_id'])}{suffix}")


@mcp.tool()
async def dptb_render_visualization(filename: str | None = None, artifact_id: str | None = None) -> CallToolResult:
    """Render existing DPTB output using native image/text content without recalculating scientific data."""
    if artifact_id:
        artifact = await _find_artifact(artifact_id)
        if not artifact.get("exists"):
            staged = await dptb_stage_artifact(artifact_id)
            artifact = (staged.structuredContent or {}).get("artifact") or artifact
        filename = str(artifact.get("workspace_path") or "")
    if not filename:
        raise RuntimeError("Provide filename or artifact_id.")
    return await dptb_preview_file(filename)


@mcp.tool()
async def dptb_export_matterviz(filename: str | None = None, artifact_id: str | None = None) -> dict[str, Any]:
    """Return the configured MatterViz URI for an existing DPTB output; no scientific data is recalculated."""
    if artifact_id:
        artifact = await _find_artifact(artifact_id)
        filename = str(artifact.get("workspace_path") or artifact.get("source_path") or "")
    if not filename:
        raise RuntimeError("Provide filename or artifact_id.")
    template = str(_load_config_data().get("matterviz_uri") or "matterviz://open?path={path}")
    return {"matterviz_uri": template.replace("{path}", quote(filename, safe="/:")), "source": filename}


@mcp.tool()
async def dptb_terminate_execution(chat_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Terminate the active DPTB Pilot run and recorded remote jobs, optionally matching an exact run id."""
    connection = _active_connection()
    return await _request(
        "POST",
        "/api/terminate-execution",
        json_body={"session_id": connection["session_id"], "chat_id": _active_chat(chat_id=chat_id), "run_id": run_id},
    )


@mcp.tool()
async def dptb_route_scientific_task(message: str, chat_id: str | None = None, timeout_ms: int = 600_000) -> dict[str, Any]:
    """Fail-closed scientific calculation entry point requiring Pilot hooks, backend capabilities, actual DPTB tools, and provenance."""
    health = await _health()
    capabilities = set(health.get("capabilities") or [])
    missing = sorted(STRICT_CAPABILITIES - capabilities)
    if missing:
        raise RuntimeError(f"Strict DPTB routing stopped: backend lacks capabilities {missing}.")
    hooks = _hook_status()
    if not hooks["config_present"] or not hooks["policy_script_present"]:
        raise RuntimeError("Strict DPTB routing stopped: plugin hook files are unavailable.")
    if not hooks["recent_scientific_activation"]:
        raise RuntimeError("Strict DPTB routing stopped: the Codex host did not provide a recent scientific-task hook activation. Trust/enable plugin hooks and retry in a new task.")
    result = await dptb_ask(message, chat_id=chat_id, timeout_ms=timeout_ms)
    run = result.get("run") or {}
    provenance = run.get("provenance") or {}
    tools = provenance.get("tools") or []
    required = {
        "run_id": run.get("run_id"),
        "backend_build_id": provenance.get("backend_build_id"),
        "engine": provenance.get("engine"),
        "tools": tools,
    }
    if run.get("status") != "completed" or not all(required.values()) or provenance.get("engine") != "dptb-pilot":
        raise RuntimeError(f"Strict DPTB routing stopped: completed result lacks required Pilot provenance ({_redact(required)}).")
    return result


@mcp.resource("dptb://file/{file_id}", mime_type="application/octet-stream")
async def dptb_file_resource(file_id: str) -> bytes | str:
    """Read a DPTB Pilot session file resource."""
    filename = _decode_file_id(file_id)
    connection = _active_connection()
    listing = await _request("GET", f"/api/files/{quote(connection['session_id'])}")
    info = next((item for item in listing.get("files", []) if str(item.get("name")) == filename), None)
    if info is None:
        raise RuntimeError(f"DPTB Pilot file does not exist: {filename}")
    if int(info.get("size") or 0) > MAX_INLINE_BYTES:
        return json.dumps({**info, "inline": False, "reason": "File exceeds inline resource limit"}, ensure_ascii=False, indent=2)
    payload, _ = await _download_bytes(f"/api/download/{quote(connection['session_id'])}/{quote(filename, safe='/')}")
    return payload


@mcp.resource("dptb://artifact/{artifact_id}", mime_type="application/octet-stream")
async def dptb_artifact_resource(artifact_id: str) -> bytes | str:
    """Read a recorded DPTB Pilot artifact resource without implicitly staging remote data."""
    artifact = await _find_artifact(artifact_id)
    if not artifact.get("exists") or not artifact.get("workspace_path"):
        return json.dumps({**artifact, "inline": False, "stage_required": bool(artifact.get("can_stage"))}, ensure_ascii=False, indent=2)
    if int(artifact.get("size") or 0) > MAX_INLINE_BYTES:
        return json.dumps({**artifact, "inline": False, "reason": "Artifact exceeds inline resource limit"}, ensure_ascii=False, indent=2)
    connection = _active_connection()
    payload, _ = await _download_bytes(
        f"/api/download/{quote(connection['session_id'])}/{quote(str(artifact['workspace_path']), safe='/')}"
    )
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
