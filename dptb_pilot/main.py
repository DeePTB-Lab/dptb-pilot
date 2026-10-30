import argparse
import os
import sys
import subprocess
import threading
import time
import webbrowser
from typing import Dict
from dotenv import load_dotenv

from dptb_pilot.core.logger import get_logger

logger = get_logger(__name__)

import shutil
import sys

def find_npm_command():
    """
    跨平台查找 npm 可执行文件
    """
    if sys.platform.startswith("win"):
        return shutil.which("npm.cmd") or shutil.which("npm.exe")
    else:
        return shutil.which("npm")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Better AIM React 启动程序")

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("PORT", 50003)),
        help="后端API服务器端口号 (默认: 50003)"
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
        default=int(os.getenv("FRONTEND_PORT", 50002)),
        help="前端开发服务器端口号 (默认: 50002)"
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
        default=None,  # Handled dynamically
        help="MCP工具服务器链接 (默认: http://{BACKEND_HOST}:{MCP_TOOLS_PORT}/sse)"
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

    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.getenv("WORK_ROOT", os.getenv("WORK_DIR", ".")),
        help="工作目录 (默认: WORK_ROOT env or 当前目录)"
    )

    return parser.parse_args()


def start_frontend_server(frontend_port: int = 3000, frontend_host: str = "0.0.0.0", backend_host: str = "localhost", backend_port: int = 8000, debug: bool = False):
    """启动前端开发服务器 (仅在开发模式下)"""
    npm_cmd = find_npm_command()
    if not npm_cmd:
        logger.error("未找到 npm，请确认 Node.js 已安装并已加入 PATH")
        return False

    # 更智能的路径查找：尝试多种可能的位置
    possible_paths = [
        # 1. Standard location: Project root/web_ui
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "web_ui"),
        # 2. Current dir/web_ui
        os.path.join(os.path.dirname(__file__), "..", "web_ui"),
        # 3. Relative path
        os.path.abspath(os.path.join(os.getcwd(), "web_ui")),
    ]

    frontend_path = None
    for path in possible_paths:
        if os.path.exists(path):
            frontend_path = path
            break

    if frontend_path is None:
        frontend_path = possible_paths[0]  # Default

    # Check for built static files
    dist_path = os.path.join(frontend_path, "dist")
    has_static = os.path.exists(dist_path) and os.path.exists(os.path.join(dist_path, "index.html"))
    
    if has_static and not debug:
        logger.info(f"✅ 发现前端编译产物，将由后端统一托管: {dist_path}")
        logger.info(f"🌍 访问地址: http://{backend_host}:{backend_port}")
        
        # 自动打开浏览器
        if backend_host in ['localhost', '127.0.0.1', '0.0.0.0']:
             target_url = f"http://localhost:{backend_port}"
             threading.Timer(2, lambda: webbrowser.open(target_url)).start()
        return True

    logger.debug(f"Current File: {__file__}")
    logger.debug(f"CWD: {os.getcwd()}")
    logger.debug(f"Tried paths: {possible_paths}")
    logger.info(f"Using Web UI path: {frontend_path}")

    if not os.path.exists(frontend_path):
        logger.error(f"前端目录不存在: {frontend_path}")
        return False

    # Fallback to npm run dev
    try:
        # 检查是否已安装依赖
        node_modules = os.path.join(frontend_path, "node_modules")
        if not os.path.exists(node_modules):
            logger.info("正在安装前端依赖...")
            try:
                result = subprocess.run(
                    [npm_cmd, "install"],
                    cwd=frontend_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("前端依赖安装完成")
                if result.stdout:
                    logger.debug(f"npm install输出: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"npm install失败: 返回码={e.returncode}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                return False

        # 启动开发服务器
        logger.info(f"启动前端开发服务器 (Dev Mode)...")
        logger.info(f"前端路径: {frontend_path}")
        logger.info(f"前端配置: {frontend_host}:{frontend_port}")
        logger.info(f"后端代理: {backend_host}:{backend_port}")

        # 设置环境变量
        env = {**os.environ}
        env.update({
            "PORT": str(frontend_port),
            "HOST": frontend_host,
            "BACKEND_HOST": backend_host,
            "BACKEND_PORT": str(backend_port)
        })

        logger.debug(f"环境变量设置: PORT={env['PORT']}, HOST={env.get('HOST', 'undefined')}, BACKEND_HOST={env.get('BACKEND_HOST', 'undefined')}, BACKEND_PORT={env.get('BACKEND_PORT', 'undefined')}")

        # 使用Popen启动，这样可以在后台运行
        logger.info("执行npm run dev...")
        process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 等待服务器启动
        logger.info("等待前端服务器启动...")
        for i in range(10):  # 等待最多10秒
            time.sleep(1)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"前端服务器启动失败: \nstdout: {stdout}\nstderr: {stderr}")
                return False

            # 尝试访问端口来检查服务器是否启动
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((frontend_host, frontend_port))
                sock.close()
                if result == 0:
                    logger.info(f"前端服务器已启动，访问地址: http://{frontend_host}:{frontend_port}")
                    # 延迟打开浏览器（仅当是localhost或127.0.0.1时）
                    if frontend_host in ['localhost', '127.0.0.1']:
                        threading.Timer(2, lambda: webbrowser.open(f"http://{frontend_host}:{frontend_port}")).start()
                    return True
            except:
                pass

        # 如果10秒后仍未启动
        stdout, stderr = process.communicate()
        logger.error(f"前端服务器启动超时: \nstdout: {stdout}\nstderr: {stderr}")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"启动前端服务器失败: {e}\n返回码: {e.returncode}\nstdout: {e.stdout}\nstderr: {e.stderr}")
        return False
    except FileNotFoundError as e:
        logger.error(f"未找到npm命令，请确保已安装Node.js和npm: {e}")
        return False
    except Exception as e:
        logger.error(f"前端服务器启动时出现未知错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def react_launch(agent_info: Dict,
                model_config: Dict,
                mcp_server_url: str = "http://0.0.0.0:50001/sse",
                work_path: str = "/tmp",
                tools_need_modify: list = None,
                host: str = "0.0.0.0",
                port: int = 50003,
                frontend_port: int = 50002,
                frontend_host: str = "0.0.0.0",
                backend_host: str = "localhost",
                no_dev: bool = False,
                debug: bool = False,
                api_key: str = None):
    """启动React版本的Better AIM"""

    # Import the server stack only after main() has loaded the startup .env
    # and resolved model_config. Dependencies may load their own dotenv files.
    from dptb_pilot.server.app import initialize_server, run_server
    from dptb_pilot.core.photon_service import init_photon_service
    from dptb_pilot.core.photon_config import PHOTON_CONFIG, CHARGING_ENABLED

    # 设置API密钥
    if api_key:
        os.environ["API_KEY"] = api_key
        model_config["api_key"] = api_key
    else:
        if os.getenv("API_KEY"):
            model_config["api_key"] = os.getenv("API_KEY")
        elif os.getenv("LLM_API_KEY"):
            model_config["api_key"] = os.getenv("LLM_API_KEY")
        else:
            logger.warning("警告: API_KEY环境变量未设置，请通过--api-key参数设置或设置环境变量")

    # 初始化光子收费服务（如果启用）
    if CHARGING_ENABLED:
        try:
            init_photon_service(PHOTON_CONFIG)
            logger.info("✅ 光子收费服务已启用")
        except Exception as e:
            logger.error(f"❌ 初始化光子收费服务失败: {e}")
            logger.warning("⚠️ 将在无光子收费模式下运行")
    else:
        logger.info("ℹ️ 光子收费服务已禁用")

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
            args=(frontend_port, frontend_host, backend_host, port, debug),  # frontend_port, frontend_host, backend_host, backend_port, debug
            daemon=True
        )
        frontend_thread.start()
        time.sleep(2)  # 等待前端服务器启动

    # 启动后端API服务器
    logger.info(f"启动后端API服务器: {host}:{port}")
    if not no_dev:
        logger.info(f"前端开发服务器: {frontend_host}:{frontend_port}")
    else:
        logger.info("前端生产模式 - 请构建前端文件并提供HTTP服务")

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("\n服务器已停止")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    # 1. 在导入服务依赖前加载启动目录的 .env；显式环境变量仍优先。
    cwd_env = os.path.join(os.getcwd(), '.env')
    if os.path.exists(cwd_env):
        logger.info(f"📄 Loading .env from current directory: {cwd_env}")
        load_dotenv(cwd_env)
    else:
        logger.info("ℹ️ No .env found in current directory, using system environment variables")

    # 2. 关键参数检查
    api_key = os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        logger.critical("❌ CRITICAL ERROR: API Key not found!")
        logger.critical("Please set LLM_API_KEY in your .env file or environment variables.")
        logger.critical(f"Expected .env path: {cwd_env}")
        sys.exit(1)

    # 3. 建议参数检查
    if not os.getenv("LLM_MODEL"):
        logger.warning("⚠️ LLM_MODEL not set, using default model")
    if not os.getenv("LLM_API_BASE"):
        logger.warning("⚠️ LLM_API_BASE not set, using default base URL")
    if not os.getenv("MP_API_KEY"):
        logger.warning("⚠️ MP_API_KEY not set, Materials Project tools will not work")

    # 4. 动态构建 MCP_TOOLS_URL (如果未设置)
    # 获取相关配置 (带默认值)
    backend_host = os.getenv("BACKEND_HOST", "localhost")
    mcp_port = os.getenv("MCP_TOOLS_PORT", "50001")
    
    # 构建默认 URL
    default_mcp_url = f"http://{backend_host}:{mcp_port}/sse"
    
    # 如果环境变量里设置了 MCP_TOOLS_URL，它会被 argparse 的 default 用 os.getenv 获取到
    # 但我们需要在这里处理 "如果没设env也没传参" 的情况，或者覆盖 argparse 的默认行为？
    # Argparse default is `os.getenv("MCP_TOOLS_URL", "http://localhost:50002/sse")`
    # Let's override the environment variable if it's missing, so argparse picks it up?
    # No, better to pass it explicitly to parse_args logic or handle it after.
    
    # 实际上 parse_arguments 里的 default 已经写死了。
    # 我们需要在调用 parse_arguments 之前或者之后处理。
    # 由于 parse_arguments 内部用了 os.getenv 作为 default，所以要在它之前 set env?
    # 或者修改 parse_arguments 的逻辑。
    
    # Let's modify parse_arguments to use this dynamic default if env is missing.
    if not os.getenv("MCP_TOOLS_URL"):
        os.environ["MCP_TOOLS_URL"] = default_mcp_url

    args = parse_arguments()

    if not args.mcp_tools:
        args.mcp_tools = os.getenv("MCP_TOOLS_URL", default_mcp_url)
    
    logger.info(f"🔗 MCP Tools URL: {args.mcp_tools}")

    # 获取绝对路径的工作目录，以便Agent能准确找到
    abs_work_dir = os.path.abspath(args.work_dir)

    # 默认的agent配置（如果没有外部配置）
    default_agent_info = {
        "name": "DeePTB-agent",
        "description": "AI agent with mcp tools for machine learning tight binding Hamiltonian predicting package DeePTB.",
        "instruction": f"""You are an expert in AI and computational materials science, specifically specializing in the DeePTB package.
Your role is twofold:
1. **Knowledge Expert**: Answer questions about DeePTB's usage, theory, and implementation.
   - You have access to the full source code and documentation in: `dptb_pilot/tools/data/deeptb_knowledge/repo`
   - You have access to relevant academic papers in: `dptb_pilot/tools/data/deeptb_knowledge/pdfs`
   - **PURE RAG WORKFLOW**:
     1. **Search Only**: You have NO access to the file system. You MUST use `search_knowledge_base` to find all information.
     2. **Trust RAG**: The knowledge base contains AST-parsed code chunks (classes/functions) and notebook cells. The search results are your ONLY source of truth.
     3. **No File Reading**: Do not attempt to use `read_file_content` or `list_directory` as they are disabled.
   - **Execution and parallel submission**:
     - Independent tool calls may be emitted in the same response so ADK can execute them concurrently.
     - In remote mode, submit independent jobs first and collect their job IDs; do not wait for one remote job before submitting another.
     - Never parallelize calls with dependent outputs, shared writable paths, or conflicting resources.
     - After all submissions, use the job status/results tools as needed and report each job separately.
   - Do not guess. Verify your answers against the search results.

2. **Execution Assistant**: Help users perform tasks like generating training configs, submitting missions, and testing models.
   - Use the available MCP tools to assist the user.
   - **File Uploads**: User uploaded files (e.g., POSCAR) are located in `{abs_work_dir}/{{session_id}}/files`.
   - **Workspace Management**:
     1. Use `list_workspace_files` to see what files are available in the workspace.
     2. If multiple structure files exist, ask the user which one to use. If only one relevant file exists, proceed to use it immediately without asking.
     3. In remote mode, if a tool result contains a Path starting with "bohrium://", immediately call `download_artifact` with:
        - `artifact_url`: the complete `bohrium://...` value.
        - `work_path`: `{abs_work_dir}/{{session_id}}/files`.
        Use the returned `file_name` in `/api/download/{{session_id}}/<file_name>`. Never claim the artifact cannot be downloaded without first checking whether `download_artifact` is available.
   - **Image Display**: You CAN display images generated by tools (like `band.png`).
     - Use this markdown format: `![Image Name](/api/download/{{session_id}}/<filename>)`
     - Example: `![Band Structure](/api/download/{{session_id}}/band.png)`

   **CRITICAL: Tool Call Formatting**
   You MUST use this EXACT format for tool calls. Do not use any other XML tags.
   
   Example:
   `<tool_calls_begin><tool_call_begin><tool_name>list_workspace_files</tool_name><parameters><work_path>/tmp/session_id/files</work_path></parameters><tool_call_end><tool_calls_end>`

   Constraints:
   - Your output MUST be exactly one single line without any newlines or spaces between tags.
   - Start immediately with `<tool_calls_begin>`.
   - IMPORTANT: `<tool_call_begin>` is NOT a standard XML tag. Do NOT close it with `</tool_call_begin>`.
   - You MUST use `<tool_call_end>` to close a tool call.
   - Use `list_workspace_files` to check files before doing anything else.
   - **Structure Visualization**:
     - If the user asks to "show", "visualize", or "display" a structure (POSCAR, CIF, etc.), use the `visualize_structure` tool.
     - **CRITICAL**: You MUST include the EXACT output of `visualize_structure` (the `:::visualize...:::` block) in your final response. Do NOT summarize it.
     - Example response: "Here is the structure: \n:::visualize\n{{...}}\n:::"
   - **Brillouin Zone Visualization**:
     - If the user asks to visualize the "Brillouin Zone", "BZ", or "k-path", use the `visualize_brillouin_zone` tool.
     - Like structure visualization, you MUST include the EXACT output in your response.
     - Like structure visualization, you MUST include the EXACT output in your response.
    - **Material Search Workflow (Consultative Mode)**:
      - When the user requests a complex material search (e.g., "Find me a semiconductor with broken symmetry"):
        1. **Analysis & Confirmation**: DO NOT call tools immediately. First, analyze the request and list the inferred criteria (e.g., "Formula: any", "Band Gap: >0.1 eV", "Symmetry: Non-centrosymmetric"). Ask the user: "Is this understanding correct?"
        2. **Planning**: After user confirmation, propose a step-by-step plan (e.g., "Step 1: Search MP...", "Step 2: Filter results...", "Step 3: Download top 3 structures").
        3. **Execution**: Execute steps one by one. Ask for permission before proceeding to the next major step (especially downloading).
        4. **Quantity Control**: Respect the user's limit on how many structures to show/download. If not specified, ask.
      - **Direct Action Mode**:
        - If the user request is simple and specific (e.g., "Download mp-149", "Show me the structure of mp-1234"), execute the tool IMMEDIATELY.
        - **CRITICAL**: If a search returns MULTIPLE results, do NOT download them automatically. You MUST list the results first and ask the user to select specific IDs (e.g., "Which one should I download?").
        - NEVER loop through a list of IDs to download them all unless the user EXPLICITLY says "Download ALL of them".
    - When calling mcp tools, do not use named submit_*** tools unless explicitly requested.
   - **System Stability**:
     - Ensure tool call tags (if used) are well-formed. Do NOT output duplicate tags like `<tool_call_end> <tool_call_end>`.
     - Only output ONE `<tool_call_end>` at the end of the tool call block.
"""
    }

    default_model_config = {
        'model': os.getenv("LLM_MODEL", "openai/qwen3-max"),
        'api_base': os.getenv("LLM_API_BASE", "https://llm.dp.tech"),
        'api_key': os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
    }

    logger.info("LLM model: %s", default_model_config["model"])

    default_tools_modify = ["band_with_baseline_model"]

    react_launch(
        agent_info=default_agent_info,
        model_config=default_model_config,
        mcp_server_url=args.mcp_tools,
        work_path=args.work_dir,
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
