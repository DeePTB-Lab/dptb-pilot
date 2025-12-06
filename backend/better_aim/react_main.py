import argparse
import os
import sys
import subprocess
import threading
import time
import webbrowser
from typing import Dict
from dotenv import load_dotenv

from better_aim.react_host import initialize_server, run_server

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Better AIM React 启动程序")

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="后端API服务器端口号 (默认: 8000)"
    )

    parser.add_argument(
        "--host", "-l",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="后端API服务器主机地址 (默认: 0.0.0.0)"
    )

    parser.add_argument(
        "--frontend-port", "-fp",
        type=int,
        default=int(os.getenv("FRONTEND_PORT", 50001)),
        help="前端开发服务器端口号 (默认: 50001)"
    )

    parser.add_argument(
        "--frontend-host",
        type=str,
        default=os.getenv("FRONTEND_HOST", "0.0.0.0"),
        help="前端开发服务器主机地址 (默认: 0.0.0.0)"
    )

    parser.add_argument(
        "--backend-host",
        type=str,
        default=os.getenv("BACKEND_HOST", "localhost"),
        help="后端API服务器主机地址 (前端代理用) (默认: localhost)"
    )

    parser.add_argument(
        "--mcp_tools",
        type=str,
        default=os.getenv("MCP_TOOLS_URL", "http://localhost:50002/sse"),
        help="MCP工具服务器链接 (默认: http://localhost:50002/sse)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API密钥 (优先级高于环境变量)"
    )

    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="不启动前端开发服务器，使用生产模式"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启调试模式"
    )

    return parser.parse_args()


def start_frontend_server(frontend_port: int = 3000, frontend_host: str = "0.0.0.0", backend_host: str = "localhost", backend_port: int = 8000):
    """启动前端开发服务器"""
    # 更智能的路径查找：尝试多种可能的位置
    possible_paths = [
        # 1. 标准位置：项目根目录下的frontend
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend"),
        # 2. 当前目录的frontend（如果在better_aim目录中运行）
        os.path.join(os.path.dirname(__file__), "..", "frontend"),
        # 3. 相对路径
        os.path.abspath(os.path.join(os.getcwd(), "frontend")),
    ]

    frontend_path = None
    for path in possible_paths:
        if os.path.exists(path):
            frontend_path = path
            break

    if frontend_path is None:
        frontend_path = possible_paths[0]  # 使用第一个作为默认值

    print(f"当前文件路径: {__file__}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试的路径: {possible_paths}")
    print(f"使用的前端路径: {frontend_path}")

    if not os.path.exists(frontend_path):
        print(f"前端目录不存在: {frontend_path}")
        return False

    try:
        # 检查是否已安装依赖
        node_modules = os.path.join(frontend_path, "node_modules")
        if not os.path.exists(node_modules):
            print("正在安装前端依赖...")
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print("前端依赖安装完成")
                if result.stdout:
                    print(f"npm install输出: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"npm install失败:")
                print(f"返回码: {e.returncode}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                return False

        # 启动开发服务器
        print(f"启动前端开发服务器...")
        print(f"前端路径: {frontend_path}")
        print(f"前端配置: {frontend_host}:{frontend_port}")
        print(f"后端代理: {backend_host}:{backend_port}")

        # 设置环境变量
        env = {**os.environ}
        env.update({
            "PORT": str(frontend_port),
            "HOST": frontend_host,
            "BACKEND_HOST": backend_host,
            "BACKEND_PORT": str(backend_port)
        })

        print(f"环境变量设置:")
        print(f"  PORT={env['PORT']}")
        print(f"  HOST={env.get('HOST', 'undefined')}")
        print(f"  BACKEND_HOST={env.get('BACKEND_HOST', 'undefined')}")
        print(f"  BACKEND_PORT={env.get('BACKEND_PORT', 'undefined')}")

        # 使用Popen启动，这样可以在后台运行
        print("执行npm run dev...")
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 等待服务器启动
        print("等待前端服务器启动...")
        for i in range(10):  # 等待最多10秒
            time.sleep(1)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"前端服务器启动失败:")
                print(f"stdout: {stdout}")
                print(f"stderr: {stderr}")
                return False

            # 尝试访问端口来检查服务器是否启动
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((frontend_host, frontend_port))
                sock.close()
                if result == 0:
                    print(f"前端服务器已启动，访问地址: http://{frontend_host}:{frontend_port}")
                    # 延迟打开浏览器（仅当是localhost或127.0.0.1时）
                    if frontend_host in ['localhost', '127.0.0.1']:
                        threading.Timer(2, lambda: webbrowser.open(f"http://{frontend_host}:{frontend_port}")).start()
                    return True
            except:
                pass

        # 如果10秒后仍未启动
        stdout, stderr = process.communicate()
        print(f"前端服务器启动超时:")
        print(f"stdout: {stdout}")
        print(f"stderr: {stderr}")
        return False

    except subprocess.CalledProcessError as e:
        print(f"启动前端服务器失败: {e}")
        print(f"返回码: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError as e:
        print(f"未找到npm命令，请确保已安装Node.js和npm: {e}")
        return False
    except Exception as e:
        print(f"前端服务器启动时出现未知错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def react_launch(agent_info: Dict,
                model_config: Dict,
                mcp_server_url: str = "http://0.0.0.0:50001/sse",
                work_path: str = "/tmp",
                tools_need_modify: list = None,
                host: str = "0.0.0.0",
                port: int = 8000,
                frontend_port: int = 50001,
                frontend_host: str = "0.0.0.0",
                backend_host: str = "localhost",
                no_dev: bool = False,
                debug: bool = False,
                api_key: str = None):
    """启动React版本的Better AIM"""

    # 设置API密钥
    if api_key:
        os.environ["API_KEY"] = api_key
        model_config["api_key"] = api_key
    else:
        if os.getenv("API_KEY"):
            model_config["api_key"] = os.getenv("API_KEY")
        else:
            print("警告: API_KEY环境变量未设置，请通过--api-key参数设置或设置环境变量")

    # 初始化后端服务器
    initialize_server(
        agent_info_dict=agent_info,
        model_config_dict=model_config,
        mcp_url=mcp_server_url,
        work_dir=work_path,
        tools_modify=tools_need_modify
    )

    # 启动前端开发服务器（如果需要）
    if not no_dev:
        frontend_thread = threading.Thread(
            target=start_frontend_server,
            args=(frontend_port, frontend_host, backend_host, port),  # frontend_port, frontend_host, backend_host, backend_port
            daemon=True
        )
        frontend_thread.start()
        time.sleep(2)  # 等待前端服务器启动

    # 启动后端API服务器
    print(f"启动后端API服务器: {host}:{port}")
    if not no_dev:
        print(f"前端开发服务器: {frontend_host}:{frontend_port}")
    else:
        print("前端生产模式 - 请构建前端文件并提供HTTP服务")

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    if load_dotenv():
        print("环境变量已根据`.env`文件读入")
    else:
        print("未找到`.env`文件或无任何变量被读入")

    args = parse_arguments()

    # 默认的agent配置（如果没有外部配置）
    default_agent_info = {
        "name": "DeePTB-agent",
        "description": "AI agent with mcp tools for machine learning tight binding Hamiltonian predicting package DeePTB.",
        "instruction": """You are an expert in AI and computational materials science, specifically specializing in the DeePTB package.
Your role is twofold:
1. **Knowledge Expert**: Answer questions about DeePTB's usage, theory, and implementation.
   - You have access to the full source code and documentation in: `backend/dptb_agent_tools/data/deeptb_knowledge/repo`
   - You have access to relevant academic papers in: `backend/dptb_agent_tools/data/deeptb_knowledge/pdfs`
   - **PURE RAG WORKFLOW**:
     1. **Search Only**: You have NO access to the file system. You MUST use `search_knowledge_base` to find all information.
     2. **Trust RAG**: The knowledge base contains AST-parsed code chunks (classes/functions) and notebook cells. The search results are your ONLY source of truth.
     3. **No File Reading**: Do not attempt to use `read_file_content` or `list_directory` as they are disabled.
   - **Sequential Execution**: Please execute tool calls ONE BY ONE.
   - Do not guess. Verify your answers against the search results.

2. **Execution Assistant**: Help users perform tasks like generating training configs, submitting missions, and testing models.
   - Use the available MCP tools to assist the user.
   - **File Uploads**: User uploaded files (e.g., POSCAR) are located in `/tmp/{session_id}`.
   - **Workspace Management**:
     1. Always set `work_path` to `/tmp/{session_id}` when calling tools.
     2. Use `list_workspace_files` to see what files are available in the workspace.
     3. If multiple structure files exist, ask the user which one to use.
   - **Image Display**: You CAN display images generated by tools (like `band.png`).
     - Use this markdown format: `![Image Name](/api/download/{session_id}/<filename>)`
     - Example: `![Band Structure](/api/download/{session_id}/band.png)`
   - **Structure Visualization**:
     - If the user asks to "show", "visualize", or "display" a structure (POSCAR, CIF, etc.), use the `visualize_structure` tool.
     - **CRITICAL**: You MUST include the EXACT output of `visualize_structure` (the `:::visualize...:::` block) in your final response. Do NOT summarize it.
     - Example response: "Here is the structure: \n:::visualize\n{...}\n:::"
   - **Brillouin Zone Visualization**:
     - If the user asks to visualize the "Brillouin Zone", "BZ", or "k-path", use the `visualize_brillouin_zone` tool.
     - Like structure visualization, you MUST include the EXACT output in your response.
   - When calling mcp tools, do not use named submit_*** tools unless explicitly requested.
"""
    }

    default_model_config = {
        'model': os.getenv("LLM_MODEL", "openai/qwen3-max"),
        'api_base': os.getenv("LLM_API_BASE", "https://llm.dp.tech"),
        'api_key': os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    }

    default_tools_modify = ["band_with_baseline_model"]

    react_launch(
        agent_info=default_agent_info,
        model_config=default_model_config,
        mcp_server_url=args.mcp_tools,
        tools_need_modify=default_tools_modify,
        host=args.host,
        port=args.port,
        frontend_port=args.frontend_port,
        frontend_host=args.frontend_host,
        backend_host=args.backend_host,
        no_dev=args.no_dev,
        debug=args.debug,
        api_key=args.api_key
    )


if __name__ == "__main__":
    main()