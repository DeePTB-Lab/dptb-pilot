# DPTB Pilot Codex Plugin

This headless plugin migrates the VS Code extension's business capabilities to native Codex MCP tools, structured results, and clickable resources. The default backend is `http://localhost:50003`.

## Capabilities

- Connection profiles, generated 32-character sessions, health checks, and capability negotiation.
- Isolated chat creation, selection, deletion, history, and persistence.
- Asynchronous FIFO tasks, wait/cancel, and parallel `depends_on` task graphs.
- Workflow plans, Pilot Review, approval/rejection, and structured parameter confirmation.
- Local, Bohrium, and Slurm profiles with execution mode and concurrency controls.
- Multi-file upload, native preview, deletion, and workspace-controlled downloads.
- Clickable `dptb://file/...` and `dptb://artifact/...` resources with remote staging.
- Native image/text display, optional MatterViz export, and Pilot token/Photon/RMB usage.

`dptb_ask` remains a compatibility wrapper and uses the shared submit-plus-wait runtime.

## Strict Scientific Routing

Materials, physics, electronic-structure, band/DOS, transport/NEGF, molecular-dynamics, and other scientific numerical calculations must use `dptb_route_scientific_task`. Default-discovered hooks mark scientific prompts and deny Python, NumPy, SciPy, pymatgen, ASE, custom formulas, or other non-DPTB compute paths.

A result is accepted only when the backend has the strict capabilities and the completed run includes a run ID, backend build ID, actual DPTB tool list, and `provenance.engine=dptb-pilot`. The plugin stops without a calculation fallback when Pilot is unavailable, unsupported, unprovenanced, or the Codex host has not trusted/activated the hooks.

## Configuration

The plugin reads `DPTB_PILOT_CONFIG`, otherwise `%USERPROFILE%\.dptb-pilot-codex\config.json`. Legacy flat configuration is migrated to v2 automatically. See [config.example.json](config.example.json).

```powershell
$env:DPTB_PILOT_SERVER_URL = "http://localhost:50003"
$env:DPTB_PILOT_SESSION_ID = "0123456789abcdef0123456789abcdef"
$env:DPTB_PILOT_TIMEOUT_MS = "30000"
$env:DPTB_PILOT_PYTHON = "C:\path\to\.venv\Scripts\python.exe"
```

Bohrium passwords are stored only in the OS credential store. `DPTB_BOHRIUM_PASSWORD` and profile-specific `DPTB_BOHRIUM_PASSWORD_<PROFILE_ID>` are supported without persisting or returning the secret.

## Install

Start the DPTB tool server and backend:

```powershell
uv sync
uv run dptb-tools
uv run dptb-pilot
```

Then run:

```powershell
cd C:\Users\lonya\PycharmProjects\dptb-pilot\codex_plugin\dptb-pilot
.\scripts\install-local.ps1 `
  -Force `
  -PythonPath "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe" `
  -ServerUrl "http://localhost:50003" `
  -OpenDeeplink
```

The installer generates a session ID when omitted and preserves/migrates existing configuration. Trust the plugin hooks during installation and start a new Codex task so the new skill and MCP definitions load.

## Validation

```powershell
uv run pytest tests/test_codex_plugin.py tests/test_server_codex_runtime.py
uv run pytest
cd vscode_extension
npm test
```

Also run the plugin and skill validators. Local development updates use the plugin-creator cachebuster/reinstall flow.
