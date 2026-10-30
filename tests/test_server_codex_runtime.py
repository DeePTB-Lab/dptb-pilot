import asyncio
import json

import pytest

from dptb_pilot.server import app as server_app


def test_runtime_batch_model_uses_independent_list_defaults():
    first = server_app.RuntimeBatchTask(id="first", message="one")
    second = server_app.RuntimeBatchTask(id="second", message="two")
    first.depends_on.append("parent")
    assert second.depends_on == []


def test_runtime_batch_endpoint_executes_independent_nodes_in_parallel(monkeypatch):
    active = 0
    peak = 0

    async def fake_run(_session_id, node):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return {"node": node.id}

    monkeypatch.setattr(server_app, "_run_prompt_batch_node", fake_run)
    server_app.batch_orchestrators.clear()
    server_app.batch_runtime_tasks.clear()
    server_app.batch_snapshots.clear()

    async def scenario():
        request = server_app.RuntimeBatchRequest(
            session_id="b" * 32,
            tasks=[
                server_app.RuntimeBatchTask(id="one", message="first"),
                server_app.RuntimeBatchTask(id="two", message="second"),
            ],
            parallel_mode="parallel",
            max_parallelism=2,
        )
        initial = await server_app.enqueue_runtime_batch(request)
        group_id = initial["task_group_id"]
        await server_app.batch_runtime_tasks[group_id]
        return server_app.batch_snapshots[group_id]

    completed = asyncio.run(scenario())
    assert peak == 2
    assert completed["status"] == "succeeded"
    assert {item["status"] for item in completed["tasks"]} == {"succeeded"}


def test_usage_summary_deduplicates_replayed_run(monkeypatch, tmp_path):
    session_id = "u" * 32
    chat_id = "chat-usage"
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    server_app.chat_runtimes.clear()

    run = {
        "run_id": "same-run",
        "prompt_id": "prompt",
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "usage_total": {"prompt_tokens": 10, "candidates_tokens": 5, "cached_tokens": 2, "reasoning_tokens": 1, "total_tokens": 15},
        "charge_result": {"photon_amount": 3, "rmb_amount": 0.25},
    }
    journal = tmp_path / session_id / "chat_runtime" / "one.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"chat_id": chat_id, "recent_runs": [run]}), encoding="utf-8")
    runtime = server_app.ChatRuntime(session_id, chat_id, recent_runs=[run], loaded=True)
    server_app.chat_runtimes[server_app.chat_state_key(session_id, chat_id)] = runtime

    result = asyncio.run(server_app.get_usage_summary(session_id, chat_id))

    assert result["usage"] == {
        "prompt_tokens": 10,
        "candidates_tokens": 5,
        "cached_tokens": 2,
        "reasoning_tokens": 1,
        "total_tokens": 15,
        "event_count": 0,
    }
    assert result["charge"] == {"photon_amount": 3, "rmb_amount": 0.25}
    assert len(result["runs"]) == 1


def test_delete_file_rejects_path_traversal(monkeypatch, tmp_path):
    session_id = "d" * 32
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    with pytest.raises(server_app.HTTPException) as exc_info:
        asyncio.run(server_app.delete_file(session_id, "../../outside.txt"))

    assert exc_info.value.status_code == 400
    assert outside.read_text(encoding="utf-8") == "keep"


def test_terminate_run_id_guard_does_not_mark_another_run(monkeypatch):
    session_id = "t" * 32
    chat_id = "chat-terminate"
    state_key = server_app.chat_state_key(session_id, chat_id)
    runtime = server_app.ChatRuntime(session_id, chat_id, loaded=True)
    runtime.active_run = server_app.RunState("active-run", "prompt", "task", 2)
    server_app.chat_runtimes[state_key] = runtime
    server_app.termination_requested.pop(state_key, None)

    request = server_app.TerminateExecutionRequest(
        session_id=session_id,
        chat_id=chat_id,
        run_id="different-run",
    )
    with pytest.raises(server_app.HTTPException) as exc_info:
        asyncio.run(server_app.terminate_execution(request))

    assert exc_info.value.status_code == 409
    assert state_key not in server_app.termination_requested
