import os

port = os.environ.get("DPTB_AGENT_PORT", "50002")
host = os.environ.get("DPTB_AGENT_HOST", "0.0.0.0")
model = os.environ.get("DPTB_AGENT_MODEL", "dp")

assert model == "dp", f"Invalid DPTB_AGENT_MODEL={model}. Please set it to dp"
from dp.agent.server import CalculationMCPServer
from dptb_pilot.tools.artifacts import register_nested_path_artifacts
from dptb_pilot.tools.executor import register_nested_path_dispatcher
from dptb_pilot.tools.mcp_resilience import register_disconnect_safe_logging

register_nested_path_artifacts()
register_nested_path_dispatcher()
register_disconnect_safe_logging()
mcp = CalculationMCPServer("DPTB_AGENT", port=port, host=host)

'''if model == "dp":
    from dp.agent.server import CalculationMCPServer
    mcp = CalculationMCPServer("DPTB_AGENT", port=port, host=host)
elif model == "fastmcp":
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("DPTB_AGENT", port=port, host=host)
elif model == "test": # For unit test of models
    class MCP:
        def tool(self):
            def decorator(func):
                return func
            return decorator
    mcp = MCP()
else:
    print("Please set the environment variable DPTB_AGENT_MODEL to dp, fastmcp or test.")
    raise ValueError("Invalid DPTB_AGENT_MODEL. Please set it to dp, fastmcp or test.")'''
