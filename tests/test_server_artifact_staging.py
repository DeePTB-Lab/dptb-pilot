import asyncio
from pathlib import Path

import pytest

from dptb_pilot.core.artifact_registry import (
    list_recorded_artifacts,
    record_completed_task_artifacts,
)
from dptb_pilot.server import app as server_app


class FakeStorage:
    calls = []

    def _download(self, key: str, destination: str) -> str:
        self.calls.append((key, destination))
        Path(destination).write_bytes(b"raw tgz bytes")
        return destination


def test_stage_artifact_downloads_exact_tgz_and_links_manifest(tmp_path, monkeypatch):
    session_id = "s" * 32
    task = record_completed_task_artifacts(
        tmp_path,
        session_id,
        "chat-1",
        "run abacus",
        r"bohrium://13756/320004/store/run/abacus\_output\_path/OUT.ABACUS.tgz",
    )
    artifact_id = task["artifacts"][0]["id"]
    FakeStorage.calls = []
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    monkeypatch.setattr(
        server_app.calculation_mcp_server,
        "init_storage",
        lambda config: ("bohrium", FakeStorage()),
    )

    response = asyncio.run(server_app.stage_artifact(session_id, artifact_id))

    artifact = response["artifact"]
    staged = tmp_path / session_id / "files" / artifact["workspace_path"]
    assert staged.name == "OUT.ABACUS.tgz"
    assert staged.read_bytes() == b"raw tgz bytes"
    assert FakeStorage.calls[0][0].endswith("abacus_output_path/OUT.ABACUS.tgz")
    assert artifact["exists"] is True
    assert list_recorded_artifacts(tmp_path, session_id)[0]["artifacts"][0]["workspace_path"] == artifact["workspace_path"]


def test_stage_artifact_rejects_unknown_record(tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    with pytest.raises(server_app.HTTPException) as exc_info:
        asyncio.run(server_app.stage_artifact("s" * 32, "missing"))

    assert exc_info.value.status_code == 404
