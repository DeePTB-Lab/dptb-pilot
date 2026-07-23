# DPTB Pilot Codex Plugin

This plugin connects Codex to an already-running DPTB Pilot FastAPI backend. It is intentionally headless: it does not embed the web UI or the VS Code webview, and instead exposes the backend as Codex-callable MCP tools.

## Tools

- `dptb_status`: check backend health and login for the configured session.
- `dptb_ask`: send a question or workflow request to the configured Pilot agent.
- `dptb_list_files`: list files in the configured session workspace.
- `dptb_upload_file`: upload a local file into the configured session.
- `dptb_download_file`: download a session file to a local path.
- `dptb_get_pending_schema`: read the current tool-parameter schema waiting for confirmation.
- `dptb_terminate_execution`: stop the active task for the configured session.

## Requirements

Prepare the repo environment and start the DPTB Pilot backend first:

```powershell
uv sync
uv run dptb-tools
uv run dptb-pilot
```

The plugin MCP adapter needs a Python runtime with `mcp` and `httpx`. The recommended runtime is the repo virtualenv:

```text
C:/Users/lonya/PycharmProjects/dptb-pilot/.venv/Scripts/python.exe
```

## Install In Codex

The Codex desktop app does not ship a `codex` CLI, so `codex plugin add ...` is not recognized there. Installation is a two-step process: run the PowerShell script to write the plugin files, config, and personal marketplace locally, then open a deeplink in the Codex desktop app to finish.

Run the installer from the plugin directory:

```powershell
cd C:\Users\lonya\PycharmProjects\dptb-pilot\codex_plugin\dptb-pilot
.\scripts\install-local.ps1 `
  -Force `
  -PythonPath "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe" `
  -ServerUrl "http://localhost:8000" `
  -SessionId "0123456789abcdef0123456789abcdef"
```

The script:

- copies the plugin to `%USERPROFILE%\plugins\dptb-pilot`
- writes connection settings to `%USERPROFILE%\.dptb-pilot-codex\config.json`
- creates or updates the personal marketplace at `%USERPROFILE%\.agents\plugins\marketplace.json`

At the end the script prints a deeplink like:

```text
codex://plugins/dptb-pilot?marketplacePath=C%3A%5CUsers%5Clonya%5C.agents%5Cplugins%5Cmarketplace.json
```

Open this link in the Codex desktop app to finish installation. Add `-OpenDeeplink` to let the script open it automatically:

```powershell
.\scripts\install-local.ps1 -Force -OpenDeeplink -PythonPath "..." -SessionId "..."
```

After installing or updating the plugin, start a new Codex thread so Codex loads the new skill and MCP tool definitions.

## Manual Configuration

Without the installer, create this config file yourself:

```json
{
  "server_url": "http://localhost:8000",
  "session_id": "0123456789abcdef0123456789abcdef",
  "timeout_ms": 30000,
  "python_path": "C:/Users/lonya/PycharmProjects/dptb-pilot/.venv/Scripts/python.exe"
}
```

Save it as:

```text
%USERPROFILE%\.dptb-pilot-codex\config.json
```

You can also point at another config or Python runtime with environment variables:

```powershell
$env:DPTB_PILOT_CONFIG = "C:\absolute\path\to\config.json"
$env:DPTB_PILOT_PYTHON = "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe"
```

Connection settings can also come directly from environment variables:

```powershell
$env:DPTB_PILOT_SERVER_URL = "http://localhost:8000"
$env:DPTB_PILOT_SESSION_ID = "0123456789abcdef0123456789abcdef"
$env:DPTB_PILOT_TIMEOUT_MS = "30000"
```

The `session_id` must be exactly 32 characters.

## Local Development

After editing the plugin, reinstall the local copy and reopen the deeplink:

```powershell
.\scripts\install-local.ps1 -Force -SkipConfig -OpenDeeplink
```

The `.mcp.json` starts a PowerShell wrapper:

```text
powershell -NoProfile -ExecutionPolicy Bypass -File ./mcp_server/run.ps1
```

The wrapper resolves Python in this order:

1. `DPTB_PILOT_PYTHON`
2. `python_path` in the config file
3. `.venv\Scripts\python.exe` under the repo or plugin directory
4. `python` on `PATH`

Once the backend is running, call `dptb_status` first to confirm the connection, then use `dptb_ask` for actual Pilot tasks.
