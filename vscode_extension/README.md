# DPTB Pilot VS Code Extension

This package adds an independent VS Code interaction mode for dptb-pilot. It connects directly to the existing FastAPI backend and does not replace or embed the existing `web_ui/` frontend.

## Features

- Configure and connect to a dptb-pilot backend URL.
- Login with a 32-character session id or generate one locally.
- Stream chat responses over `/ws/chat/{session_id}`.
- Manage chat sessions using the existing `/api/user/{user_id}/sessions` protocol.
- Upload files from VS Code or browser file picker.
- List, preview, download, and delete session files.
- Render Markdown, images, token/charge metadata, and `:::visualize` blocks with a JSON fallback.
- Render structured `:::visualize` payloads for structure, BZ, curve, image, and table views.
- Hand off supported visualizations to MatterViz through a configurable command id or URI template.
- Confirm tool parameters using form or JSON mode.
- Select Local/Remote execution mode and manage Bohrium/Slurm machine JSON configs.
- Bind the selected remote executor to the active Pilot session automatically.
- Open via command or `vscode://` deep link.

## Install in VS Code / 在 VS Code 中安装

This folder can be built into a normal VS Code `.vsix` extension package and installed directly. The dptb-pilot FastAPI backend must be running separately before you connect from the extension; the default URL is `http://localhost:50003`.

本目录可以打包成标准的 VS Code `.vsix` 插件并直接安装。连接前需要先单独启动 dptb-pilot FastAPI 后端；默认地址为 `http://localhost:50003`。

From this directory:

```bash
npm install
npm run compile
npm run package
```

The package command generates a file such as:

```text
dptb-pilot-vscode-0.1.0.vsix
```

Install it with either method:

- VS Code UI: open the Extensions view, click `...`, choose `Install from VSIX...`, and select `dptb-pilot-vscode-0.1.0.vsix`.
- Command line:

  ```bash
  code --install-extension dptb-pilot-vscode-0.1.0.vsix --force
  ```

After installation, reload VS Code if prompted, then use the command palette:

- `DPTB Pilot: Open Chat`
- `DPTB Pilot: Connect Server`
- `DPTB Pilot: Upload Current File`

If `code` is not available in your shell, run `Shell Command: Install 'code' command in PATH` from VS Code's command palette, or use the VS Code UI installation method above.

## Development

For development and debugging, from this directory:

```bash
npm install
npm run compile
npm test
```

Open this folder in VS Code and run the extension in an Extension Development Host. Use the command palette:

- `DPTB Pilot: Open Chat`
- `DPTB Pilot: Connect Server`
- `DPTB Pilot: Upload Current File`

## Configuration

Settings:

- `dptbPilot.serverUrl` — defaults to `http://localhost:50003`.
- `dptbPilot.sessionId` — optional default session id.
- `dptbPilot.autoOpenOnUrl` — controls whether deep links open the chat view.
- `dptbPilot.matterViz.commandId` — optional MatterViz command used to open a visualization payload.
- `dptbPilot.matterViz.uriTemplate` — optional URI template with `{payload}` placeholder.
- `dptbPilot.matterViz.autoOpen` — automatically forward supported visualizations when enabled.

Deep link format:

```text
vscode://dptb-pilot.dptb-pilot-vscode/connect?url=http%3A%2F%2Flocalhost%3A50003&session=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef
```

## Backend Requirements

Start the existing dptb-pilot backend first. The extension uses the existing endpoints implemented in `dptb_pilot/server/app.py`:

- `POST /api/login`
- `POST /api/execution-config`
- `GET /api/config`
- `GET/POST /api/user/{user_id}/sessions`
- `GET /api/files/{session_id}`
- `POST /api/upload/{session_id}`
- `GET /api/download/{session_id}/{filename}`
- `DELETE /api/files/{session_id}/{filename}`
- `GET /api/schema/{session_id}`
- `POST /api/modify-params`
- `POST /api/terminate-execution`
- `WebSocket /ws/chat/{session_id}`

## Manual Verification

1. Start the dptb-pilot backend at `http://localhost:50003` (or use the value configured by `PORT`).
2. Launch the extension development host.
3. Run `DPTB Pilot: Connect Server` and enter the backend URL.
4. Generate or enter a 32-character session id and connect.
5. Send a chat message and verify streaming output.
6. Upload a file and verify list/preview/download/delete.
7. Trigger a tool requiring parameters and verify form/json submission and termination.
8. Select Remote mode and a complete Bohrium or Slurm machine, then verify a
   compute tool is submitted remotely without adding an `executor` argument to
   the chat request. The backend generates the SDK-compatible executor object;
   `"bohrium"` is not a valid executor shorthand.
9. Test the deep link above with your own URL/session.
10. Confirm the original `web_ui/` frontend still runs independently.

## Notes

The first implementation intentionally keeps the VS Code package independent. Shared types are copied into `src/shared/types.ts`; if both frontends evolve together, these can later be moved to a shared workspace package.
