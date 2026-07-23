---
name: use-dptb-pilot
description: Use when the user asks Codex to call DPTB Pilot, upload files to a Pilot session, list Pilot files, or ask the configured DeePTB agent to help with a workflow.
---

# Use DPTB Pilot

Use the DPTB Pilot MCP tools when the user wants help from the configured DPTB Pilot backend.

## Configuration

The plugin reads connection settings from `DPTB_PILOT_CONFIG`, then from
`~/.dptb-pilot-codex/config.json`, then from environment variables:

- `DPTB_PILOT_SERVER_URL`
- `DPTB_PILOT_SESSION_ID`
- `DPTB_PILOT_TIMEOUT_MS`

The `session_id` must be exactly 32 characters.

## Tool Routing

- Use `dptb_status` before the first real action when connection health is uncertain.
- Use `dptb_ask` for agent questions or workflow requests.
- Use `dptb_upload_file` before asking about a local structure or input file.
- Use `dptb_list_files` to inspect session workspace outputs.
- Use `dptb_download_file` when the user asks to bring a generated file back into the local workspace.
- Use `dptb_get_pending_schema` if an agent response indicates tool parameters need confirmation.
- Use `dptb_terminate_execution` only when the user asks to stop the current Pilot task.
