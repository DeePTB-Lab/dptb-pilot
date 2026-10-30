from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from dptb_pilot.tools.artifacts import handle_nested_input_artifacts


class FakeStorage:
    calls = []

    def download(self, key: str, destination: str) -> str:
        self.calls.append((key, destination))
        return str(Path(destination) / Path(key).name)


def _patch_storage(monkeypatch):
    FakeStorage.calls = []
    monkeypatch.setattr(
        'dp.agent.server.calculation_mcp_server.init_storage',
        lambda config: ('local', FakeStorage()),
    )


def test_optional_path_list_is_downloaded_recursively(monkeypatch):
    _patch_storage(monkeypatch)

    def tool(pp_orb_paths: Optional[List[Path]] = None):
        pass

    kwargs, artifacts = handle_nested_input_artifacts(
        tool,
        {'pp_orb_paths': ['C.upf', 'C.orb']},
        None,
    )

    assert kwargs['pp_orb_paths'] == [
        Path('inputs/pp_orb_paths/0/C.upf'),
        Path('inputs/pp_orb_paths/1/C.orb'),
    ]
    assert artifacts['pp_orb_paths']['uri'] == ['C.upf', 'C.orb']


def test_union_and_nested_path_annotations_are_supported(monkeypatch):
    _patch_storage(monkeypatch)

    def tool(
        files: Union[List[Path], List[str], None] = None,
        grouped: Optional[Dict[str, list[Path]]] = None,
        checkpoint: Path | None = None,
    ):
        pass

    kwargs, _ = handle_nested_input_artifacts(
        tool,
        {
            'files': ['Si.upf'],
            'grouped': {'orb': ['Si.orb']},
            'checkpoint': 'model.pt',
        },
        None,
    )

    assert kwargs['files'] == [Path('inputs/files/0/Si.upf')]
    assert kwargs['grouped'] == {
        'orb': [Path('inputs/grouped/value-0/0/Si.orb')]
    }
    assert kwargs['checkpoint'] == Path('inputs/checkpoint/model.pt')


def test_non_path_collections_are_untouched(monkeypatch):
    _patch_storage(monkeypatch)

    def tool(extra_cli_args: Optional[list[str]] = None):
        pass

    value = ['--restart', 'model.pt']
    kwargs, artifacts = handle_nested_input_artifacts(
        tool,
        {'extra_cli_args': value},
        None,
    )

    assert kwargs['extra_cli_args'] is value
    assert artifacts == {}


def test_iterable_paths_and_windows_absolute_paths(monkeypatch, tmp_path):
    _patch_storage(monkeypatch)

    def tool(files: Iterable[Path]):
        pass

    windows_path = r'C:\workspace\Si.upf'
    kwargs, _ = handle_nested_input_artifacts(
        tool,
        {'files': [windows_path]},
        None,
    )

    assert kwargs['files'] == [Path('inputs/files/0/Si.upf')]
    assert FakeStorage.calls == [(windows_path, str(Path('inputs/files/0')))]
