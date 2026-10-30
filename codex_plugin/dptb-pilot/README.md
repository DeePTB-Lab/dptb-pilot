# DPTB Pilot Codex 插件

该插件将 VS Code 扩展的业务能力迁移为 Codex 原生 MCP 工具、结构化结果和可点击资源，不维护独立 Webview。默认连接 `http://localhost:50003`。

English documentation: [README_en.md](README_en.md)

## 功能

- 连接 profile、32 位 session 自动生成、健康与 capability 协商。
- 多 chat 创建、选择、删除、历史隔离与持久化。
- 异步单任务、FIFO 队列、等待/取消，以及带 `depends_on` 的并行任务图。
- 工作流计划、Pilot Review、批准/拒绝和结构化参数确认。
- Local、Bohrium、Slurm profile；并行模式和最大并发数设置。
- 多文件上传、预览、删除和受控目录下载。
- `dptb://file/...` 与 `dptb://artifact/...` 可点击资源，远程 artifact staging。
- 原生图片/文本展示、MatterViz 可选导出，以及 Pilot token/Photon/RMB 汇总。

`dptb_ask` 仅作为兼容入口，内部使用 `dptb_submit_task` + `dptb_wait_run`。

## 严格科学计算策略

材料、物理、电子结构、能带/DOS、输运/NEGF、分子动力学和其他科学数值计算必须使用 `dptb_route_scientific_task`。插件通过默认发现的 `hooks/hooks.json`：

- 在 `UserPromptSubmit` 标记科学任务；
- 在 `PreToolUse` 阻止 Python、NumPy、SciPy、pymatgen、ASE、自定义公式和其他非 DPTB 计算工具；
- 要求后端具有严格 capability；
- 仅接受包含 `run_id`、backend build ID、实际 DPTB tool 列表和 `provenance.engine=dptb-pilot` 的完成结果。

Pilot 离线、能力不足、hook 未被主机信任/触发或 provenance 不完整时直接停止，不做计算回退。插件安装后需要在 Codex 中信任 hooks，并新建任务加载新工具。

## 配置

配置读取顺序：`DPTB_PILOT_CONFIG` 指定文件，否则 `%USERPROFILE%\.dptb-pilot-codex\config.json`。旧扁平配置会自动升级为 v2。示例见 [config.example.json](config.example.json)。

可覆盖的环境变量：

```powershell
$env:DPTB_PILOT_SERVER_URL = "http://localhost:50003"
$env:DPTB_PILOT_SESSION_ID = "0123456789abcdef0123456789abcdef"
$env:DPTB_PILOT_TIMEOUT_MS = "30000"
$env:DPTB_PILOT_PYTHON = "C:\path\to\.venv\Scripts\python.exe"
```

Bohrium 密码只保存到操作系统凭据库。也可使用 `DPTB_BOHRIUM_PASSWORD` 或 `DPTB_BOHRIUM_PASSWORD_<PROFILE_ID>`，不会写入配置、日志或 MCP 返回。

## 安装

先准备依赖并启动工具服务与主后端：

```powershell
uv sync
uv run dptb-tools
uv run dptb-pilot
```

本地首次安装：

```powershell
cd C:\Users\lonya\PycharmProjects\dptb-pilot\codex_plugin\dptb-pilot
.\scripts\install-local.ps1 `
  -Force `
  -PythonPath "C:\Users\lonya\PycharmProjects\dptb-pilot\.venv\Scripts\python.exe" `
  -ServerUrl "http://localhost:50003" `
  -OpenDeeplink
```

未提供 `SessionId` 时安装器自动生成 32 位 ID；已有 v1/v2 配置会被保留并迁移。脚本复制插件、写入个人 marketplace，并打开安装 deeplink。

## 开发验证

```powershell
uv run pytest tests/test_codex_plugin.py tests/test_server_codex_runtime.py
uv run pytest
cd vscode_extension
npm test
```

插件与 skill 还应分别通过 `validate_plugin.py` 和 `quick_validate.py`。本地更新使用 plugin-creator 的 cachebuster/reinstall 流程，更新后在 Codex 新建任务验收。
