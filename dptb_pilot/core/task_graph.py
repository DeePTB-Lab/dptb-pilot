"""Validation and scheduling primitives for safe parallel task submission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


TERMINAL_STATUSES = {"succeeded", "failed", "blocked", "cancelled"}


@dataclass(frozen=True)
class TaskNode:
    """A declarative executable workflow node."""

    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    parallel_safe: bool = False
    read_resources: frozenset[str] = frozenset()
    write_resources: frozenset[str] = frozenset()
    submit_policy: str = "wait"
    retry_policy: str = "none"

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int = 0) -> "TaskNode":
        if not isinstance(value, dict):
            raise ValueError(f"task {index or '?'} must be an object")
        task_id = str(value.get("id") or f"T{index or 1}").strip()
        tool = str(value.get("tool") or "").strip()
        if not task_id:
            raise ValueError("task id must not be empty")
        if not tool:
            raise ValueError(f"task {task_id} must define a tool")

        def string_set(key: str) -> frozenset[str]:
            raw = value.get(key) or []
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise ValueError(f"task {task_id} field {key} must be a list")
            return frozenset(str(item).strip() for item in raw if str(item).strip())

        raw_deps = value.get("depends_on") or []
        if isinstance(raw_deps, (str, int)):
            raw_deps = [raw_deps]
        if not isinstance(raw_deps, (list, tuple, set, frozenset)):
            raise ValueError(f"task {task_id} field depends_on must be a list")

        args = value.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"task {task_id} args must be an object")
        return cls(
            id=task_id,
            tool=tool,
            args=args,
            depends_on=tuple(str(item).strip() for item in raw_deps if str(item).strip()),
            parallel_safe=bool(value.get("parallel_safe", False)),
            read_resources=string_set("read_resources"),
            write_resources=string_set("write_resources"),
            submit_policy=str(value.get("submit_policy") or "wait"),
            retry_policy=str(value.get("retry_policy") or "none"),
        )


def _conflicts(left: TaskNode, right: TaskNode) -> bool:
    return bool(
        left.write_resources
        & (right.write_resources | right.read_resources)
        or right.write_resources
        & (left.write_resources | left.read_resources)
    )


def validate_task_graph(tasks: Iterable[TaskNode]) -> dict[str, TaskNode]:
    """Validate IDs, dependencies and cycles, returning an indexed graph."""
    graph: dict[str, TaskNode] = {}
    for task in tasks:
        if task.id in graph:
            raise ValueError(f"duplicate task id: {task.id}")
        graph[task.id] = task

    for task in graph.values():
        missing = set(task.depends_on) - set(graph)
        if missing:
            raise ValueError(f"task {task.id} depends on unknown task(s): {sorted(missing)}")
        if task.id in task.depends_on:
            raise ValueError(f"task {task.id} cannot depend on itself")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"task graph contains a cycle at {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id].depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)
    return graph


def select_ready_tasks(
    graph: dict[str, TaskNode],
    statuses: dict[str, str],
    active: Iterable[str] = (),
) -> list[TaskNode]:
    """Return dependency-ready nodes in stable declaration order."""
    active_ids = set(active)
    ready = []
    for task in graph.values():
        if statuses.get(task.id, "queued") != "queued" or task.id in active_ids:
            continue
        dependency_statuses = [statuses.get(dep, "queued") for dep in task.depends_on]
        if any(status in {"failed", "blocked", "cancelled"} for status in dependency_statuses):
            continue
        if all(status == "succeeded" for status in dependency_statuses):
            ready.append(task)
    return ready


def choose_parallel_batch(
    ready: Iterable[TaskNode],
    max_parallelism: int,
) -> list[TaskNode]:
    """Choose a conflict-free batch; unsafe nodes are intentionally serialized."""
    limit = max(1, int(max_parallelism))
    selected: list[TaskNode] = []
    for task in ready:
        if len(selected) >= limit:
            break
        if not task.parallel_safe and selected:
            continue
        if not task.parallel_safe:
            selected = [task]
            break
        if all(not _conflicts(task, other) for other in selected):
            selected.append(task)
    return selected


def normalize_task_graph(plan: dict[str, Any]) -> dict[str, TaskNode]:
    """Read the new ``tasks`` shape, with a compatibility path for steps."""
    raw_tasks = plan.get("tasks") if isinstance(plan, dict) else None
    if raw_tasks is None and isinstance(plan, dict):
        raw_tasks = plan.get("steps")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("workflow plan must contain a non-empty tasks list")

    normalized: list[TaskNode] = []
    for index, raw in enumerate(raw_tasks, 1):
        raw = dict(raw)
        if "tool" not in raw:
            tools = raw.get("tools") or []
            if isinstance(tools, str):
                tools = [tools]
            if len(tools) != 1:
                raise ValueError(
                    f"workflow task {raw.get('id') or index} must define exactly one executable tool"
                )
            raw["tool"] = tools[0]
        normalized.append(TaskNode.from_dict(raw, index))
    return validate_task_graph(normalized)
