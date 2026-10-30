import asyncio
import json

from dptb_pilot.server import app as server_app


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def accept(self):
        return None

    async def send_text(self, value):
        self.messages.append(value)


def test_runtime_worker_executes_queued_prompts_in_fifo_order(monkeypatch):
    executed = []
    events = []

    async def fake_execute(_runtime, _run, prompt, _cookies):
        executed.append(prompt.prompt_id)
        return {"status": "completed"}

    async def fake_persist(_runtime):
        return None

    async def fake_append(*_args, **_kwargs):
        return None

    async def fake_send(_session_id, message):
        events.append(message)

    monkeypatch.setattr(server_app, "_execute_chat_prompt_limited", fake_execute)
    monkeypatch.setattr(server_app, "_persist_chat_runtime", fake_persist)
    monkeypatch.setattr(server_app, "_append_persisted_messages", fake_append)
    monkeypatch.setattr(server_app.manager, "send_message", fake_send)

    runtime = server_app.ChatRuntime("session", "chat", loaded=True)
    runtime.queue.put_nowait(server_app.QueuedPrompt("p1", "first", protocol_version=2))
    runtime.queue.put_nowait(server_app.QueuedPrompt("p2", "second", protocol_version=2))

    asyncio.run(server_app._runtime_worker(runtime, {}))

    assert executed == ["p1", "p2"]
    assert [event["type"] for event in events if event["type"].startswith("run_")] == [
        "run_started", "run_completed", "run_started", "run_completed",
    ]
    assert runtime.active_run is None
    assert runtime.queue.empty()


def test_completed_stages_are_ordered_and_idempotently_identified(monkeypatch):
    persisted = []
    events = []

    async def fake_append(_session_id, _chat_id, messages):
        persisted.extend(messages)

    async def fake_persist(_runtime):
        return None

    async def fake_send(_session_id, message):
        events.append(message)

    monkeypatch.setattr(server_app, "_append_persisted_messages", fake_append)
    monkeypatch.setattr(server_app, "_persist_chat_runtime", fake_persist)
    monkeypatch.setattr(server_app.manager, "send_message", fake_send)
    runtime = server_app.ChatRuntime("session", "chat", loaded=True)
    run = server_app.RunState("run", "prompt", "request", 2)

    async def scenario():
        await server_app._append_run_stage(runtime, run, stage_id="one", content="first")
        await server_app._append_run_stage(runtime, run, stage_id="two", content="second")

    asyncio.run(scenario())

    assert [message["content"] for message in persisted] == ["first", "second"]
    assert [message["sequence"] for message in persisted] == [1, 2]
    assert len({message["message_id"] for message in persisted}) == 2
    assert [event["type"] for event in events] == ["stage_completed", "stage_completed"]


def test_tool_results_are_sent_and_persisted_with_the_stage(monkeypatch):
    persisted = []
    events = []

    async def fake_append(_session_id, _chat_id, messages):
        persisted.extend(messages)

    async def fake_persist(_runtime):
        return None

    async def fake_send(_session_id, message):
        events.append(message)

    monkeypatch.setattr(server_app, "_append_persisted_messages", fake_append)
    monkeypatch.setattr(server_app, "_persist_chat_runtime", fake_persist)
    monkeypatch.setattr(server_app.manager, "send_message", fake_send)
    runtime = server_app.ChatRuntime("session", "chat", loaded=True)
    run = server_app.RunState("run", "prompt", "request", 2)
    tool_results = [{"tool": "inspect", "input": {"path": "a.json"}, "result": {"ok": True}}]

    asyncio.run(server_app._append_run_stage(
        runtime,
        run,
        stage_id="tool-batch-1",
        content="Tool batch completed: inspect",
        tools=["inspect"],
        tool_results=tool_results,
    ))

    assert events[0]["tool_results"] == tool_results
    assert persisted[0]["tool_results"] == tool_results


def test_removing_a_queued_prompt_does_not_touch_the_active_run(monkeypatch):
    async def fake_persist(_runtime):
        return None

    async def fake_send(_session_id, _message):
        return None

    monkeypatch.setattr(server_app, "_persist_chat_runtime", fake_persist)
    monkeypatch.setattr(server_app.manager, "send_message", fake_send)
    runtime = server_app.ChatRuntime("session", "chat", loaded=True)
    runtime.active_run = server_app.RunState("run", "active", "active", 2)
    runtime.queue.put_nowait(server_app.QueuedPrompt("p1", "first", protocol_version=2))
    runtime.queue.put_nowait(server_app.QueuedPrompt("p2", "second", protocol_version=2))

    removed = asyncio.run(server_app._remove_queued_prompt(runtime, "p1"))

    assert removed is True
    assert runtime.active_run.prompt_id == "active"
    assert [item.prompt_id for item in server_app._queued_prompts(runtime)] == ["p2"]


def test_runtime_journal_restores_queue_and_marks_active_run_interrupted(monkeypatch, tmp_path):
    session_id = "R" * 32
    chat_id = "chat-reconnect"
    sessions_file = tmp_path / session_id / "sessions.json"
    sessions_file.parent.mkdir(parents=True)
    sessions_file.write_text(json.dumps([{
        "chat_id": chat_id,
        "history": [],
    }]), encoding="utf-8")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    server_app.chat_runtimes.clear()

    runtime = server_app.ChatRuntime(session_id, chat_id, loaded=True)
    runtime.queue.put_nowait(server_app.QueuedPrompt("queued", "next", protocol_version=2))
    runtime.active_run = server_app.RunState("active-run", "active", "current", 2)
    server_app._write_runtime_journal(runtime)
    server_app.chat_runtimes.clear()

    restored = asyncio.run(server_app.get_chat_runtime(session_id, chat_id))

    assert restored.active_run is None
    assert [item.prompt_id for item in server_app._queued_prompts(restored)] == ["queued"]
    assert restored.recent_runs[-1]["status"] == "interrupted"
    history = json.loads(sessions_file.read_text(encoding="utf-8"))[0]["history"]
    assert history[-1]["status"] == "interrupted"


def test_connection_manager_routes_events_to_the_matching_chat_only():
    async def scenario():
        manager = server_app.ConnectionManager()
        first = FakeWebSocket()
        second = FakeWebSocket()
        await manager.connect(first, "session", "chat-one")
        await manager.connect(second, "session", "chat-two")
        await manager.send_message("session", {"type": "stage_completed", "chat_id": "chat-one"})
        return first, second

    first, second = asyncio.run(scenario())
    assert len(first.messages) == 1
    assert second.messages == []
