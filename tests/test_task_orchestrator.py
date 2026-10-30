import asyncio
import time

import pytest

from dptb_pilot.core.orchestrator import TaskOrchestrator
from dptb_pilot.core.task_graph import (
    TaskNode,
    choose_parallel_batch,
    normalize_task_graph,
)


def test_graph_rejects_cycles_and_unknown_dependencies():
    with pytest.raises(ValueError, match="unknown"):
        normalize_task_graph({"tasks": [{"id": "a", "tool": "x", "depends_on": ["missing"]}]})
    with pytest.raises(ValueError, match="cycle"):
        normalize_task_graph({"tasks": [
            {"id": "a", "tool": "x", "depends_on": ["b"]},
            {"id": "b", "tool": "y", "depends_on": ["a"]},
        ]})


def test_conflicting_writes_are_not_selected_together():
    tasks = [
        TaskNode("a", "x", parallel_safe=True, write_resources=frozenset({"out"})),
        TaskNode("b", "y", parallel_safe=True, write_resources=frozenset({"out"})),
        TaskNode("c", "z", parallel_safe=True, write_resources=frozenset({"other"})),
    ]
    selected = choose_parallel_batch(tasks, 3)
    assert [task.id for task in selected] == ["a", "c"]


def test_orchestrator_runs_independent_tasks_concurrently_and_dependencies_afterwards():
    async def scenario():
        started = {}
        finished = {}

        async def invoke(node):
            started[node.id] = time.monotonic()
            await asyncio.sleep(0.05)
            finished[node.id] = time.monotonic()
            return {"task": node.id}

        plan = {"execution": {"max_parallelism": 2}, "tasks": [
            {"id": "a", "tool": "one", "parallel_safe": True},
            {"id": "b", "tool": "two", "parallel_safe": True},
            {"id": "c", "tool": "three", "depends_on": ["a", "b"]},
        ]}
        result = await TaskOrchestrator(plan, invoke).run()
        assert result["status"] == "succeeded"
        assert result["tasks"][0]["idempotency_key"].startswith(result["task_group_id"] + ":")
        assert result["tasks"][0]["status"] == "succeeded"
        assert started["c"] >= max(finished["a"], finished["b"])
        assert abs(started["a"] - started["b"]) < 0.03

    asyncio.run(scenario())


def test_orchestrator_blocks_descendants_after_failure():
    async def scenario():
        async def invoke(node):
            if node.id == "bad":
                raise RuntimeError("boom")
            return {"ok": node.id}

        result = await TaskOrchestrator({"tasks": [
            {"id": "bad", "tool": "x"},
            {"id": "child", "tool": "y", "depends_on": ["bad"]},
        ]}, invoke).run()
        statuses = {item["id"]: item["status"] for item in result["tasks"]}
        assert result["status"] == "failed"
        assert statuses == {"bad": "failed", "child": "blocked"}

    asyncio.run(scenario())


def test_orchestrator_cancellation_marks_queued_and_running_tasks():
    async def scenario():
        started = asyncio.Event()

        async def invoke(_node):
            started.set()
            await asyncio.sleep(10)

        orchestrator = TaskOrchestrator({"tasks": [
            {"id": "a", "tool": "x", "parallel_safe": True},
            {"id": "b", "tool": "y", "parallel_safe": True},
        ]}, invoke)
        running = asyncio.create_task(orchestrator.run())
        await started.wait()
        await orchestrator.cancel()
        result = await running
        assert result["status"] == "cancelled"
        assert all(item["status"] == "cancelled" for item in result["tasks"])

    asyncio.run(scenario())
