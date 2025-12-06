from pathlib import Path
import importlib
import os
import argparse
from dotenv import load_dotenv

from importlib.metadata import version
__version__ = version("dptb_pilot_backend")

def load_tools():
    """
    Load all tools from the dptb_agent_tools package.
    """
    # The original dynamic loading mechanism is replaced by explicit imports
    # based on the provided "Code Edit" which lists specific modules.
    # The instruction is to "Import visualize_tool module", and the snippet
    # shows it alongside other explicit imports.
    import dptb_agent_tools.modules.sk_baseline_model
    import dptb_agent_tools.modules.sk_predict
    import dptb_agent_tools.modules.config_tool
    import dptb_agent_tools.modules.workspace_tool
    import dptb_agent_tools.modules.visualize_tool
    import dptb_agent_tools.modules.visualize_bz_tool
    
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
        print("Address:", address + "/sse")
    elif os.environ["DPTB_AGENT_TRANSPORT"] == "streamable-http":
        print("Address:", address + "/mcp")
    else:
        raise ValueError("Invalid transport protocol specified. Use 'sse' or 'streamable-http'.")

def print_version():
    """
    Print the version of the Dptb_Agent.
    """
    repo = "nowhere"
    print(f"\nDptb_Agent Tools Version: {__version__}")
    print(f"For more information, visit: {repo}\n")

def main():
    """
    Main function to run the MCP tool.
    """
    print_version()
    if load_dotenv():
        print("✅ Environment variables loaded from .env")
    else:
        print("⚠️ .env file not found or empty")
        
    print(f"MCP_TOOLS_PORT: {os.getenv('MCP_TOOLS_PORT', 'Not Set (using default)')}")
    args = parse_args()  
    
    from dptb_agent_tools.env import set_envs, create_workpath
    set_envs(
        transport_input=args.transport,
        model_input=args.model,
        port_input=args.port, 
        host_input=args.host)
    create_workpath()

    from dptb_agent_tools.init_mcp import mcp
    load_tools()  

    print_address()
    mcp.run(transport=os.environ["DPTB_AGENT_TRANSPORT"])

if __name__ == "__main__":
    main()
