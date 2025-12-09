from pathlib import Path
import importlib
import os
import argparse
import sys
from dotenv import load_dotenv
from dptb_pilot.core.logger import get_logger

logger = get_logger(__name__)

from importlib.metadata import version
__version__ = version("dptb_pilot")

def load_tools():
    """
    Load all tools from the dptb_agent_tools package.
    """
    # The original dynamic loading mechanism is replaced by explicit imports
    # based on the provided "Code Edit" which lists specific modules.
    # The instruction is to "Import visualize_tool module", and the snippet
    # Visualization
    from dptb_pilot.tools.modules.visualization.visualize_tool import visualize_structure
    from dptb_pilot.tools.modules.visualization.visualize_bz_tool import visualize_brillouin_zone
    # from dptb_pilot.tools.modules.visualization.band import band_structure_plot

    # DeepTB
    from dptb_pilot.tools.modules.deeptb.config_tool import generate_deeptb_e3_training_config
    from dptb_pilot.tools.modules.deeptb.sk_baseline_model import band_with_baseline_model, generate_sk_baseline_model
    from dptb_pilot.tools.modules.deeptb.sk_predict import band_with_sk_model

    # Knowledge
    from dptb_pilot.tools.modules.knowledge.mp_tool import search_materials_project, download_mp_structure
    from dptb_pilot.tools.modules.knowledge.rag_tool import search_knowledge_base

    # System
    from dptb_pilot.tools.modules.system.workspace_tool import list_workspace_files, read_file_content
    
    # The following lines were part of the original dynamic loading loop
    # and are now commented out or removed as they are no longer applicable
    # with explicit imports.
    # module_dir = Path(__file__).parent / "modules"
    
    # for py_file in module_dir.glob("*.py"):
    #     if py_file.name.startswith("_") or py_file.stem in ["utils", "comm"]: 
    #         continue  # skip __init__.py and utils.py
        
    #     module_name = f"dptb_agent_tools.modules.{py_file.stem}"
    #     try:
    #         module = importlib.import_module(module_name)
    #         print(f"✅ Successfully loaded: {module_name}")
    #     except Exception as e:
    #         print(f"⚠️ Failed to load {module_name}: {str(e)}")


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Dptb Agent Command Line Interface")
    
    parser.add_argument(
        "--transport",
        type=str,
        default=None,
        choices=["sse", "streamable-http"],
        help="Transport protocol to use (default: sse), choices: sse, streamable-http"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fastmcp",
        choices=["fastmcp", "test", "dp"],
        help="Model to use (default: fastmcp), choices: fastmcp, test, dp"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_TOOLS_PORT", 50002)),
        help="Port to run the MCP server on (default: 50002)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to run the MCP server on (default: localhost)"
    )
    
    args = parser.parse_args()
    
    return args

def print_address():
    """
    Print the address of the MCP server based on environment variables.
    """
    address = f"{os.environ['DPTB_AGENT_HOST']}:{os.environ['DPTB_AGENT_PORT']}"
    if os.environ["DPTB_AGENT_TRANSPORT"] == "sse":
        logger.info(f"Address: {address}/sse")
    elif os.environ["DPTB_AGENT_TRANSPORT"] == "streamable-http":
        logger.info(f"Address: {address}/mcp")
    else:
        raise ValueError("Invalid transport protocol specified. Use 'sse' or 'streamable-http'.")

def print_version():
    """
    Print the version of the Dptb_Agent.
    """
    repo = "https://github.com/DeePTB-Lab/dptb-pilot"
    logger.info(f"\nDeePTB Agent Tools Version: {__version__}")
    logger.info(f"For more information, visit: {repo}\n")

def main():
    """
    Main function to run the MCP tool.
    """
    print_version()
def main():
    """
    Main function to run the MCP tool.
    """
    print_version()
    
    # 1. 优先加载当前运行目录下的 .env
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
        sys.exit(1)
        
    logger.debug(f"MCP_TOOLS_PORT: {os.getenv('MCP_TOOLS_PORT', 'Not Set (using default)')}")
    args = parse_args()  
    
    from dptb_pilot.tools.env import set_envs, create_workpath
    set_envs(
        transport_input=args.transport,
        model_input=args.model,
        port_input=args.port, 
        host_input=args.host)
    create_workpath()

    from dptb_pilot.tools.init import mcp
    load_tools()  

    print_address()
    mcp.run(transport=os.environ["DPTB_AGENT_TRANSPORT"])

if __name__ == "__main__":
    main()
