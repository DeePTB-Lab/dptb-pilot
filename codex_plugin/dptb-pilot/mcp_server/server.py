from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("dptb-pilot")


class DptbPilotConfig:
    def __init__(self, server_url: str, session_id: str, timeout_ms: int = 30000):
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.timeout_ms = timeout_ms


_login_lock = asyncio.Lock()
_logged_in: set[tuple[str, str]] = set()


def _config_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("DPTB_PILOT_CONFIG")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path("~/.dptb-pilot-codex/config.json").expanduser())
    return candidates


def _load_config() -> DptbPilotConfig:
    data: dict[str, Any] = {}
    for path in _config_candidates():
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            break

    server_url = os.getenv("DPTB_PILOT_SERVER_URL") or data.get("server_url")
    session_id = os.getenv("DPTB_PILOT_SESSION_ID") or data.get("session_id")
    timeout_ms = int(os.getenv("DPTB_PILOT_TIMEOUT_MS") or data.get("timeout_ms") or 30000)

    if not server_url:
        raise RuntimeError(
            "DPTB Pilot server_url is not configured. Set DPTB_PILOT_CONFIG or "
            "~/.dptb-pilot-codex/config.json."
        )
    if not session_id:
        raise RuntimeError(
            "DPTB Pilot session_id is not configured. Set DPTB_PILOT_CONFIG or "
            "~/.dptb-pilot-codex/config.json."
        )
    if len(session_id) != 32:
        raise RuntimeError("DPTB Pilot session_id must be exactly 32 characters.")

    return DptbPilotConfig(server_url=server_url, session_id=session_id, timeout_ms=timeout_ms)


def _url(config: DptbPilotConfig, path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{config.server_url}{path}"


async def _request(
    method: str,
    path: str,
    *,
    json_body: Any | None = None,
    files: Any | None = None,
    login: bool = True,
) -> Any:
    config = _load_config()
    if login:
        await _ensure_login(config)

    timeout = httpx.Timeout(config.timeout_ms / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            _url(config, path),
            json=json_body,
            files=files,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()


async def _ensure_login(config: DptbPilotConfig) -> None:
    key = (config.server_url, config.session_id)
    if key in _logged_in:
        return

    async with _login_lock:
        if key in _logged_in:
            return

        timeout = httpx.Timeout(config.timeout_ms / 1000)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _url(config, "/api/login"),
                json={"session_id": config.session_id},
            )
            response.raise_for_status()
        _logged_in.add(key)


@mcp.tool()
async def dptb_status() -> dict[str, Any]:
    """Check the configured DPTB Pilot backend and session login."""
    config = _load_config()
    timeout = httpx.Timeout(config.timeout_ms / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        health = await client.get(_url(config, "/api/health"))
        health.raise_for_status()
        await _ensure_login(config)
        app_config = await client.get(_url(config, "/api/config"))
        app_config.raise_for_status()

    return {
        "server_url": config.server_url,
        "session_id_prefix": config.session_id[:4],
        "health": health.json(),
        "config": app_config.json(),
    }


@mcp.tool()
async def dptb_ask(message: str, chat_id: str | None = None) -> dict[str, Any]:
    """Send a message to the configured DPTB Pilot agent and return its final response."""
    config = _load_config()
    payload: dict[str, Any] = {
        "session_id": config.session_id,
        "message": message,
    }
    if chat_id:
        payload["chat_id"] = chat_id

    try:
        return await _request("POST", "/api/chat", json_body=payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            _logged_in.discard((config.server_url, config.session_id))
            await _ensure_login(config)
            return await _request("POST", "/api/chat", json_body=payload, login=False)
        raise


@mcp.tool()
async def dptb_list_files() -> dict[str, Any]:
    """List files in the configured DPTB Pilot session workspace."""
    config = _load_config()
    return await _request("GET", f"/api/files/{config.session_id}")


@mcp.tool()
async def dptb_upload_file(local_path: str, remote_name: str | None = None) -> dict[str, Any]:
    """Upload a local file into the configured DPTB Pilot session workspace."""
    path = Path(local_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"File does not exist: {path}")

    config = _load_config()
    filename = remote_name or path.name
    with path.open("rb") as f:
        files = {"files": (filename, f, "application/octet-stream")}
        return await _request("POST", f"/api/upload/{config.session_id}", files=files)


@mcp.tool()
async def dptb_download_file(filename: str, output_path: str) -> dict[str, Any]:
    """Download a DPTB Pilot session file to a local output path."""
    config = _load_config()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    await _ensure_login(config)
    timeout = httpx.Timeout(config.timeout_ms / 1000)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(_url(config, f"/api/download/{config.session_id}/{filename}"))
        response.raise_for_status()
        output.write_bytes(response.content)

    return {
        "filename": filename,
        "output_path": str(output),
        "bytes": output.stat().st_size,
    }


@mcp.tool()
async def dptb_get_pending_schema() -> dict[str, Any]:
    """Get the current DPTB Pilot tool-parameter schema waiting for confirmation."""
    config = _load_config()
    return await _request("GET", f"/api/schema/{config.session_id}")


@mcp.tool()
async def dptb_terminate_execution() -> dict[str, Any]:
    """Ask DPTB Pilot to terminate the active execution for the configured session."""
    config = _load_config()
    return await _request(
        "POST",
        "/api/terminate-execution",
        json_body={"session_id": config.session_id},
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
