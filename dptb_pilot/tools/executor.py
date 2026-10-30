import shutil
import uuid
from pathlib import Path
from typing import Any

from dp.agent.server.executor import executor_dict
from dp.agent.server.executor.dispatcher_executor import DispatcherExecutor


class NestedPathDispatcherExecutor(DispatcherExecutor):
    """Forward Path values nested in list/dict tool arguments."""

    def submit(self, fn, kwargs):
        cwd = Path.cwd().resolve()
        staged_root = cwd / f".dptb_nested_inputs_{uuid.uuid4().hex}"
        forwarded: dict[Path, Path] = {}
        staged_paths: dict[Path, Path] = {}

        def prepare(value: Any, *, nested: bool = False) -> Any:
            if isinstance(value, Path):
                if not nested:
                    return value
                source = value.resolve(strict=True)
                try:
                    relative = source.relative_to(cwd)
                except ValueError:
                    relative = None
                if relative is not None and relative.parts:
                    forwarded[cwd / relative.parts[0]] = cwd / relative.parts[0]
                    return relative

                if source in staged_paths:
                    return staged_paths[source].relative_to(cwd)
                staged_path = staged_root / str(len(staged_paths)) / source.name
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, staged_path)
                else:
                    shutil.copy2(source, staged_path)
                staged_paths[source] = staged_path
                forwarded[staged_root] = staged_root
                return staged_path.relative_to(cwd)
            if isinstance(value, list):
                return [prepare(item, nested=True) for item in value]
            if isinstance(value, tuple):
                return tuple(prepare(item, nested=True) for item in value)
            if isinstance(value, set):
                return {prepare(item, nested=True) for item in value}
            if isinstance(value, frozenset):
                return frozenset(prepare(item, nested=True) for item in value)
            if isinstance(value, dict):
                return {
                    prepare(key, nested=True): prepare(item, nested=True)
                    for key, item in value.items()
                }
            return value

        prepared_kwargs = {
            key: prepare(value)
            for key, value in kwargs.items()
        }
        added_packages = [str(path) for path in forwarded.values()]
        self.python_packages.extend(added_packages)
        try:
            return super().submit(fn, prepared_kwargs)
        finally:
            for package in added_packages:
                try:
                    self.python_packages.remove(package)
                except ValueError:
                    pass
            if staged_root.exists():
                shutil.rmtree(staged_root)

    def get_results(self, job_id):
        result = super().get_results(job_id)
        if result == {} and not Path("results.txt").is_file():
            raise RuntimeError(
                f"Remote job {job_id} finished without results.txt or an error file"
            )
        return result


def register_nested_path_dispatcher() -> None:
    executor_dict['dispatcher'] = NestedPathDispatcherExecutor
