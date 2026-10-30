import asyncio

from anyio import BrokenResourceError
from mcp.server.fastmcp.server import Context

from dptb_pilot.tools.mcp_resilience import register_disconnect_safe_logging


def test_disconnected_client_does_not_break_context_logging(monkeypatch):
    async def broken_log(self, *args, **kwargs):
        raise BrokenResourceError

    monkeypatch.setattr(Context, "log", broken_log)
    register_disconnect_safe_logging()

    context = Context()
    assert asyncio.run(context.log(level="info", message="done")) is None
