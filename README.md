# DPTB Pilot

DPTB Pilot 是一个集成了 AI Agent 和 DeePTB 工具的智能辅助系统。本项目包含前端 React 应用和后端 Python 服务。

## 目录结构

*   `frontend/`: React 前端应用
*   `backend/`: Python 后端服务
    *   `better_aim/`: 主应用逻辑和 API 服务
    *   `dptb_agent_tools/`: MCP 工具集合

## 快速开始

要运行整个系统，你需要打开两个终端窗口，分别启动 **MCP 工具服务** 和 **主应用**。

### 前置条件

*   Python 3.10+
*   Node.js & npm
*   已激活 Python 虚拟环境 (推荐: `pybaim`)

### 步骤 1: 启动 MCP 工具服务 (终端 1)

这个服务提供 AI Agent 所需的工具集。为了避免端口冲突，我们将其运行在 `50002` 端口。

```bash
# 1. 激活虚拟环境
source /Users/aisiqg/Software/venv/pybaim/bin/activate

# 2. 设置 PYTHONPATH (确保能找到 backend 目录下的包)
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# 3. 启动服务
python backend/dptb_agent_tools/main.py --port 50002
```

启动成功后，你会看到类似 `Uvicorn running on http://localhost:50002` 的日志。

### 步骤 2: 启动主应用 (终端 2)

这个命令会同时启动后端 API 服务 (8000端口) 和前端开发服务器 (50001端口)。

```bash
# 1. 激活虚拟环境
source /Users/aisiqg/Software/venv/pybaim/bin/activate

# 2. 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend

# 3. 启动主应用 (指向刚才启动的 MCP 工具服务)
python backend/better_aim/react_main.py --mcp_tools http://0.0.0.0:50002/sse
```

启动成功后，会自动尝试打开浏览器。

## 访问地址

*   **前端界面**: [http://0.0.0.0:50001](http://0.0.0.0:50001) (主要访问入口)
*   **后端 API**: http://0.0.0.0:8000
*   **MCP 工具**: http://localhost:50002/sse
