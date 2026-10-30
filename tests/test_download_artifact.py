from pathlib import Path

from dptb_pilot.tools.artifacts import handle_nested_input_artifacts
from dptb_pilot.tools.modules.util import download_artifact as artifact_module


class FakeStorage:
    calls = []

    def download(self, key: str, destination: str) -> str:
        self.calls.append((key, destination))
        output = Path(destination) / Path(key).name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"remote-png-data")
        return str(output)


def test_download_artifact_stages_resolved_file_in_workspace(tmp_path):
    source = tmp_path / "resolved" / "band.png"
    source.parent.mkdir()
    source.write_bytes(b"png-data")
    workspace = tmp_path / "session" / "files"

    result = artifact_module.download_artifact(
        source,
        str(workspace),
    )

    assert result == {
        "file_path": str((workspace / "band.png").resolve()),
        "file_name": "band.png",
    }
    assert (workspace / "band.png").read_bytes() == b"png-data"


def test_download_artifact_uses_configured_storage_adapter(monkeypatch, tmp_path):
    FakeStorage.calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dp.agent.server.calculation_mcp_server.init_storage",
        lambda config: ("bohrium", FakeStorage()),
    )

    kwargs, artifacts = handle_nested_input_artifacts(
        artifact_module.download_artifact,
        {"artifact_url": "bohrium://project/outputs/band.png", "work_path": "files"},
        {"type": "bohrium", "token": "session-token"},
    )
    result = artifact_module.download_artifact(**kwargs)

    assert FakeStorage.calls == [
        ("project/outputs/band.png", str(Path("inputs/artifact_url")))
    ]
    assert artifacts["artifact_url"]["uri"] == "bohrium://project/outputs/band.png"
    assert Path(result["file_path"]).read_bytes() == b"remote-png-data"


def test_utility_loader_registers_download_artifact():
    from dptb_pilot.tools.init import mcp
    from dptb_pilot.tools.server import load_utility_tools

    load_utility_tools()

    assert "download_artifact" in mcp.mcp._tool_manager._tools
