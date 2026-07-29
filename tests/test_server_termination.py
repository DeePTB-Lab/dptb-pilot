import asyncio

from dptb_pilot.server import app as server_app


def setup_function():
    server_app.pending_events.clear()
    server_app.cancel_execution_events.clear()
    server_app.termination_requested.clear()


def test_terminate_execution_replaces_legacy_none_cancel_event():
    session_id = "session-with-stale-cancel-event"
    pending_event = asyncio.Event()
    server_app.pending_events[session_id] = pending_event
    server_app.cancel_execution_events[session_id] = None

    response = asyncio.run(
        server_app.terminate_execution(
            server_app.TerminateExecutionRequest(session_id=session_id)
        )
    )

    assert response["status"] == "terminating"
    assert pending_event.is_set()
    assert server_app.cancel_execution_events[session_id].is_set()


def test_terminate_execution_treats_none_pending_event_as_inactive():
    session_id = "session-with-stale-pending-event"
    server_app.pending_events[session_id] = None
    server_app.cancel_execution_events[session_id] = None

    response = asyncio.run(
        server_app.terminate_execution(
            server_app.TerminateExecutionRequest(session_id=session_id)
        )
    )

    assert response["status"] == "no_active_task"
    assert session_id not in server_app.pending_events
    assert session_id not in server_app.cancel_execution_events
