# DPTB Pilot

DPTB Pilot 是一个集成了 AI Agent 和 DeePTB 工具的智能辅助系统。本项目包含前端 React 应用和后端 Python 服务。

## 目录结构

*   `frontend/`: React 前端应用
*   `backend/`: Python 后端服务
    *   `better_aim/`: 主应用逻辑和 API 服务
    *   `dptb_agent_tools/`: MCP 工具集合

## 配置

```env
# LLM Configuration
LLM_MODEL=XXX
LLM_API_BASE= XXX
LLM_API_KEY=your_actual_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=50001
BACKEND_HOST=localhost
MCP_TOOLS_PORT=50002
MCP_TOOLS_URL=http://localhost:${MCP_TOOLS_PORT}/sse
```

## 快速开始

要运行整个系统，你需要打开两个终端窗口，分别启动 **MCP 工具服务** 和 **主应用**。

### 前置条件

*   Python 3.10+
*   Node.js & npm
*   已激活 Python 虚拟环境 (推荐: `pybaim`)

### 步骤 1: 安装项目

在项目根目录下执行：

```bash
pip install -e .
```

这会安装两个命令行工具：`dptb-pilot` (主应用) 和 `dptb-tools` (MCP 工具服务)。

### 步骤 2: 启动服务

您需要打开两个终端窗口。

**终端 1: 启动 MCP 工具服务**

```bash
dptb-tools
```

**终端 2: 启动主应用**

```bash
dptb-pilot
```

启动成功后，会自动尝试打开浏览器。

## 注意事项

### 代理设置 (VPN)

如果您开启了 VPN 或系统代理，可能会导致无法连接到 `localhost` 或 `127.0.0.1`。此时请设置 `NO_PROXY` 环境变量：

```bash
export NO_PROXY="localhost,127.0.0.1"
```

或者在启动命令前加上：

```bash
NO_PROXY="localhost,127.0.0.1" dptb-pilot
```

## 访问地址

*   **前端界面**: [http://0.0.0.0:50001](http://0.0.0.0:50001) (主要访问入口)
*   **后端 API**: http://0.0.0.0:8000
*   **MCP 工具**: http://localhost:50002/sse
source /Users/aisiqg/Software/venv/pybaim/bin/activate && pip uninstall -y dptb_pilot_backend better_aim dptb_agent_tools && pip install -e .