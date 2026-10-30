import logging

from anyio import BrokenResourceError, ClosedResourceError
from mcp.server.fastmcp.server import Context


logger = logging.getLogger(__name__)
_PATCH_MARKER = "_dptb_disconnect_safe_log"


def register_disconnect_safe_logging() -> None:
    """Treat logging to a disconnected MCP client as a dropped notification."""
    if getattr(Context.log, _PATCH_MARKER, False):
        return

    original_log = Context.log

    async def disconnect_safe_log(self, *args, **kwargs):
        try:
            return await original_log(self, *args, **kwargs)
        except (BrokenResourceError, ClosedResourceError):
            logger.debug("Skipped MCP log notification because the client disconnected")
            return None

    setattr(disconnect_safe_log, _PATCH_MARKER, True)
    Context.log = disconnect_safe_log
