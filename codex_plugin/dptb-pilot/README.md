# DPTB Pilot Codex 插件

这个插件把已经运行的 DPTB Pilot FastAPI 后端接入 Codex。插件本身是 headless 的，不嵌入 Web UI 或 VS Code webview，而是通过 MCP 工具让 Codex 直接访问 Pilot 会话。

English instructions are available in [README_en.md](README_en.md)。

## 提供的工具

- `dptb_status`：检查后端健康状态，并确认当前 session 可以登录。
- `dptb_ask`：向配置好的 DPTB Pilot agent 发送问题或任务请求。
- `dptb_list_files`：列出当前 session 工作区中的文件。
- `dptb_upload_file`：上传本地文件到当前 DPTB Pilot session。
- `dptb_download_file`：把 DPTB Pilot session 中的文件下载到本地。
- `dptb_get_pending_schema`：读取当前等待确认的工具参数 schema。
- `dptb_terminate_execution`：终止当前 session 中正在执行的任务。

## 运行要求

先在本仓库准备好 Python 环境，并启动 DPTB Pilot 后端：

```powershell
uv sync
uv run dptb-tools
uv run dptb-pilot
```

插件的 MCP 适配器需要能使用包含 `mcp` 和 `httpx` 的 Python。推荐使用本仓库虚拟环境：

```text
C:/Users/lonya/PycharmProjects/dptb-pilot/.venv/Scripts/python.exe
```

## 安装到 Codex

Codex 桌面端没有 `codex` 命令行工具，所以不能用 `codex plugin add`。安装流程分为两步：先运行 PowerShell 脚本把插件文件、配置和个人 marketplace 写到本地，然后在 Codex 桌面端打开 deeplink 完成安装。

从插件目录运行安装脚本：

```powershell
cd C:\Users\lonya\PycharmProjects\dptb-pilot\codex_plugin\dptb-pilot
.\scripts\install-local.ps1 `
  -Force `
  -PythonPath "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe" `
  -ServerUrl "http://localhost:8000" `
  -SessionId "0123456789abcdef0123456789abcdef"
```

脚本会完成三件事：

- 复制插件到 `%USERPROFILE%\plugins\dptb-pilot`
- 写入连接配置到 `%USERPROFILE%\.dptb-pilot-codex\config.json`
- 创建或更新个人 marketplace：`%USERPROFILE%\.agents\plugins\marketplace.json`

脚本结尾会打印一个 deeplink，形如：

```text
codex://plugins/dptb-pilot?marketplacePath=C%3A%5CUsers%5Clonya%5C.agents%5Cplugins%5Cmarketplace.json
```

在 Codex 桌面端打开这个链接即可完成安装。也可以加 `-OpenDeeplink` 让脚本自动打开：

```powershell
.\scripts\install-local.ps1 -Force -OpenDeeplink -PythonPath "..." -SessionId "..."
```

安装或更新插件后，开启一个新的 Codex 线程来测试新功能和 MCP 工具。

## 手动配置

如果不使用安装脚本，也可以手动创建配置文件：

```json
{
  "server_url": "http://localhost:8000",
  "session_id": "0123456789abcdef0123456789abcdef",
  "timeout_ms": 30000,
  "python_path": "C:/Users/lonya/PycharmProjects/dptb-pilot/.venv/Scripts/python.exe"
}
```

推荐保存到：

```text
%USERPROFILE%\.dptb-pilot-codex\config.json
```

也可以通过环境变量指定：

```powershell
$env:DPTB_PILOT_CONFIG = "C:\absolute\path\to\config.json"
$env:DPTB_PILOT_PYTHON = "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe"
```

连接参数也支持直接使用环境变量：

```powershell
$env:DPTB_PILOT_SERVER_URL = "http://localhost:8000"
$env:DPTB_PILOT_SESSION_ID = "0123456789abcdef0123456789abcdef"
$env:DPTB_PILOT_TIMEOUT_MS = "30000"
```

`session_id` 必须是后端要求的 32 位字符串。

## 本地开发

修改插件后，重新运行安装脚本并再次打开 deeplink：

```powershell
.\scripts\install-local.ps1 -Force -SkipConfig -OpenDeeplink
```

`.mcp.json` 会启动 PowerShell wrapper：

```text
powershell -NoProfile -ExecutionPolicy Bypass -File ./mcp_server/run.ps1
```

wrapper 会按顺序查找 Python：

1. `DPTB_PILOT_PYTHON`
2. 配置文件中的 `python_path`
3. 仓库或插件目录下的 `.venv\Scripts\python.exe`
4. `PATH` 上的 `python`

配置完成并启动后端后，可以先让 Codex 调用 `dptb_status`，确认连接正常后再使用 `dptb_ask` 发起实际任务。
