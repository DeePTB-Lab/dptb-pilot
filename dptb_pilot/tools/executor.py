from pathlib import Path
from typing import Any

from dp.agent.server.executor import executor_dict
from dp.agent.server.executor.dispatcher_executor import DispatcherExecutor


class NestedPathDispatcherExecutor(DispatcherExecutor):
    """Forward Path values nested in list/dict tool arguments."""

    def submit(self, fn, kwargs):
        forwarded: dict[str, Path] = {}

        def prepare(value: Any, *, nested: bool = False) -> Any:
            if isinstance(value, Path):
                if not nested:
                    return value
                source = value.resolve()
                name = source.name
                previous = forwarded.get(name)
                if previous and previous != source:
                    raise ValueError(
                        f"Remote inputs contain different files named {name}"
                    )
                forwarded[name] = source
                return Path(name)
            if isinstance(value, list):
                return [prepare(item, nested=True) for item in value]
            if isinstance(value, tuple):
                return tuple(prepare(item, nested=True) for item in value)
            if isinstance(value, dict):
                return {
                    key: prepare(item, nested=True)
                    for key, item in value.items()
                }
            return value

        prepared_kwargs = {
            key: prepare(value)
            for key, value in kwargs.items()
        }
        self.python_packages.extend(str(path) for path in forwarded.values())
        return super().submit(fn, prepared_kwargs)


def register_nested_path_dispatcher() -> None:
    executor_dict['dispatcher'] = NestedPathDispatcherExecutor
