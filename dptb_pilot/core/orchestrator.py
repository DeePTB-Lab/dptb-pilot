"""Bounded, observable execution of independent workflow tasks."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .task_graph import (
    TERMINAL_STATUSES,
    TaskNode,
    choose_parallel_batch,
    normalize_task_graph,
    select_ready_tasks,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    id: str
    tool: str
    status: str = "queued"
    attempts: int = 0
    result: Any = None
    error: str | None = None
    remote_task_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    idempotency_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "status": self.status,
            "attempts": self.attempts,
            "result": self.result,
            "error": self.error,
            "remote_task_id": self.remote_task_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "idempotency_key": self.idempotency_key,
        }


@dataclass
class TaskGroup:
    id: str
    max_parallelism: int
    failure_policy: str = "continue_independent"
    status: str = "queued"
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    cancel_requested: bool = False
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_group_id": self.id,
            "status": self.status,
            "max_parallelism": self.max_parallelism,
            "failure_policy": self.failure_policy,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "tasks": [record.as_dict() for record in self.tasks.values()],
        }


TaskInvoker = Callable[[TaskNode], Any | Awaitable[Any]]
TaskUpdate = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class TaskOrchestrator:
    """Execute a validated task graph with bounded, conflict-free parallelism."""

    def __init__(
        self,
        plan: dict[str, Any],
        invoker: TaskInvoker,
        *,
        task_group_id: str | None = None,
        max_parallelism: int | None = None,
        failure_policy: str | None = None,
        on_update: TaskUpdate | None = None,
    ) -> None:
        execution = plan.get("execution") if isinstance(plan, dict) else {}
        execution = execution if isinstance(execution, dict) else {}
        graph = normalize_task_graph(plan)
        configured_limit = max_parallelism or execution.get("max_parallelism", 4)
        self.graph = graph
        self.invoker = invoker
        self.on_update = on_update
        self.group = TaskGroup(
            id=task_group_id or uuid.uuid4().hex,
            max_parallelism=max(1, int(configured_limit)),
            failure_policy=str(failure_policy or execution.get("failure_policy") or "continue_independent"),
            tasks={task_id: TaskRecord(task_id, node.tool) for task_id, node in graph.items()},
        )
        for record in self.group.tasks.values():
            record.idempotency_key = f"{self.group.id}:{record.id}"
        self._cancel_event = asyncio.Event()
        self._running: dict[str, asyncio.Task[Any]] = {}

    async def _notify(self) -> None:
        if self.on_update is None:
            return
        value = self.group.snapshot()
        result = self.on_update(value)
        if inspect.isawaitable(result):
            await result

    async def cancel(self) -> None:
        self.group.cancel_requested = True
        self._cancel_event.set()
        for task in self._running.values():
            task.cancel()
        await self._notify()

    async def _invoke(self, node: TaskNode) -> Any:
        if inspect.iscoroutinefunction(self.invoker):
            return await self.invoker(node)
        result = await asyncio.to_thread(self.invoker, node)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _remote_id(result: Any) -> str | None:
        if not isinstance(result, dict):
            return None
        for key in ("remote_task_id", "job_id", "task_id"):
            if result.get(key):
                return str(result[key])
        extra = result.get("extra_info")
        if isinstance(extra, dict) and extra.get("bohr_job_id"):
            return str(extra["bohr_job_id"])
        return None

    async def _run_one(self, node: TaskNode) -> None:
        record = self.group.tasks[node.id]
        record.status = "submitting"
        record.started_at = _now()
        record.attempts += 1
        await self._notify()
        try:
            result = await self._invoke(node)
            record.result = result
            record.remote_task_id = self._remote_id(result)
            record.status = "submitted" if node.submit_policy == "submit_only" else "succeeded"
        except asyncio.CancelledError:
            record.status = "cancelled"
            raise
        except Exception as exc:  # task failures are isolated from siblings
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            record.finished_at = _now()
            await self._notify()

    def _block_unreachable(self) -> bool:
        changed = False
        for node_id, node in self.graph.items():
            record = self.group.tasks[node_id]
            if record.status != "queued":
                continue
            if any(self.group.tasks[dep].status in {"failed", "blocked", "cancelled"} for dep in node.depends_on):
                record.status = "blocked"
                record.finished_at = _now()
                changed = True
        return changed

    async def run(self) -> dict[str, Any]:
        self.group.status = "running"
        await self._notify()
        try:
            while True:
                if self._cancel_event.is_set():
                    for record in self.group.tasks.values():
                        if record.status in {"queued", "submitting"}:
                            record.status = "cancelled"
                            record.finished_at = _now()
                    self.group.status = "cancelled"
                    break

                self._block_unreachable()
                if all(record.status in TERMINAL_STATUSES | {"submitted"} for record in self.group.tasks.values()):
                    self.group.status = "succeeded" if all(
                        record.status in {"succeeded", "submitted"} for record in self.group.tasks.values()
                    ) else "failed"
                    break

                ready = select_ready_tasks(
                    self.graph,
                    {task_id: record.status for task_id, record in self.group.tasks.items()},
                    self._running,
                )
                batch = choose_parallel_batch(ready, self.group.max_parallelism)
                if not batch:
                    if self._running:
                        await asyncio.gather(*self._running.values(), return_exceptions=True)
                        self._running.clear()
                        continue
                    # This should only be possible for malformed state; avoid a hot loop.
                    for record in self.group.tasks.values():
                        if record.status == "queued":
                            record.status = "blocked"
                            record.error = "No executable dependency-ready task"
                    self.group.status = "failed"
                    break

                self._running = {
                    node.id: asyncio.create_task(self._run_one(node))
                    for node in batch
                }
                await asyncio.gather(*self._running.values(), return_exceptions=True)
                self._running.clear()

                if self.group.failure_policy == "fail_fast" and any(
                    record.status == "failed" for record in self.group.tasks.values()
                ):
                    for record in self.group.tasks.values():
                        if record.status == "queued":
                            record.status = "cancelled"
                    self.group.status = "failed"
                    break
        finally:
            self.group.finished_at = _now()
            await self._notify()
        return self.group.snapshot()
