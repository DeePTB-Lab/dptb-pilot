import json
import pytest

from dptb_pilot.core.artifact_registry import (
    extract_artifacts,
    get_recorded_artifact,
    link_recorded_artifact,
    list_recorded_artifacts,
    record_completed_task_artifacts,
)


SESSION_ID = "a" * 32


def test_completed_task_artifacts_are_saved_and_enriched(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files" / "results"
    files_dir.mkdir(parents=True)
    artifact_file = files_dir / "band.csv"
    artifact_file.write_text("k,e\n0,1.5\n", encoding="utf-8")

    task = record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "calculate the band structure",
        f"Result: [band data](/api/download/{SESSION_ID}/results/band.csv)",
    )

    assert task is not None
    manifest = json.loads((tmp_path / SESSION_ID / "artifacts.json").read_text(encoding="utf-8"))
    assert manifest["tasks"][0]["artifacts"][0]["workspace_path"] == "results/band.csv"

    listed = list_recorded_artifacts(tmp_path, SESSION_ID)
    artifact = listed[0]["artifacts"][0]
    assert artifact["exists"] is True
    assert artifact["preview_kind"] == "text"
    assert "0,1.5" in artifact["content_preview"]


def test_bohrium_reference_is_preserved_and_mapped_to_staged_file(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "result.zip").write_bytes(b"PK")

    artifacts = extract_artifacts(
        "Remote artifact: bohrium://bucket/jobs/42/result.zip",
        tmp_path,
        SESSION_ID,
    )

    assert artifacts == [
        {
            "id": artifacts[0]["id"],
            "source_path": "bohrium://bucket/jobs/42/result.zip",
            "workspace_path": "result.zip",
        }
    ]


def test_remote_artifact_can_be_linked_after_on_demand_staging(tmp_path):
    task = record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "remote calculation",
        r"bohrium://13756/320004/store/run/abacus\_output\_path/OUT.ABACUS.tgz",
    )
    artifact_id = task["artifacts"][0]["id"]
    staged = tmp_path / SESSION_ID / "files" / "artifacts" / artifact_id / "OUT.ABACUS.tgz"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"raw tgz")

    assert get_recorded_artifact(tmp_path, SESSION_ID, artifact_id)["workspace_path"] is None
    link_recorded_artifact(
        tmp_path,
        SESSION_ID,
        artifact_id,
        f"artifacts/{artifact_id}/OUT.ABACUS.tgz",
    )

    artifact = list_recorded_artifacts(tmp_path, SESSION_ID)[0]["artifacts"][0]
    assert artifact["exists"] is True
    assert artifact["can_stage"] is False
    assert artifact["workspace_path"] == f"artifacts/{artifact_id}/OUT.ABACUS.tgz"


def test_unrelated_web_links_are_not_recorded(tmp_path):
    assert extract_artifacts(
        "Documentation: [DeePTB](https://example.com/docs)",
        tmp_path,
        SESSION_ID,
    ) == []


def test_read_only_tool_references_are_not_recorded_as_run_artifacts(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "old-result.csv").write_text("old", encoding="utf-8")

    task = record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "inspect the workspace",
        "I inspected old-result.csv.",
        ignored_artifact_text='{"files": ["old-result.csv"]}',
    )

    assert task is None


def test_producer_output_overrides_a_read_only_reference_filter(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "updated.csv").write_text("new", encoding="utf-8")

    task = record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "update the data",
        "The result is updated.csv.",
        artifact_text='{"output_file": "updated.csv"}',
        ignored_artifact_text='{"files": ["updated.csv"]}',
    )

    assert task is not None
    assert task["artifacts"][0]["workspace_path"] == "updated.csv"


def test_relative_numpy_paths_from_structured_tool_output_are_recorded(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files" / "outputs"
    files_dir.mkdir(parents=True)
    (files_dir / "bands.npy").write_bytes(b"numpy")
    (tmp_path / SESSION_ID / "files" / "archive.npz").write_bytes(b"numpy-zip")

    artifacts = extract_artifacts(
        '{"file_path": "outputs/bands.npy", "archive": "archive.npz"}',
        tmp_path,
        SESSION_ID,
    )

    assert {item["workspace_path"] for item in artifacts} == {"outputs/bands.npy", "archive.npz"}


def test_numpy_artifacts_have_readable_metadata_preview(tmp_path):
    np = pytest.importorskip("numpy")
    files_dir = tmp_path / SESSION_ID / "files"
    files_dir.mkdir(parents=True)
    np.save(files_dir / "bands.npy", np.zeros((2, 3), dtype=np.float32))
    np.savez(files_dir / "archive.npz", energies=np.ones(4))

    # Previews are checked through the same enrichment path used by the API.
    record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "numpy outputs",
        "bands.npy archive.npz",
    )
    listed = list_recorded_artifacts(tmp_path, SESSION_ID)
    previews = {item["file_name"]: item["content_preview"] for item in listed[0]["artifacts"]}
    assert "shape=(2, 3)" in previews["bands.npy"]
    assert "energies" in previews["archive.npz"]


def test_trailing_path_separator_does_not_make_npz_unavailable(tmp_path):
    files_dir = tmp_path / SESSION_ID / "files" / "band_16_0_full"
    files_dir.mkdir(parents=True)
    (files_dir / "bandstructure_1788249247.npz").write_bytes(b"npz")

    task = record_completed_task_artifacts(
        tmp_path,
        SESSION_ID,
        "chat-1",
        "band structure",
        "/personal/dptb-pilot/workspace/"
        + SESSION_ID
        + "/files/band_16_0_full/bandstructure_1788249247.npz\\",
    )

    assert task is not None
    artifact = list_recorded_artifacts(tmp_path, SESSION_ID)[0]["artifacts"][0]
    assert artifact["workspace_path"] == "band_16_0_full/bandstructure_1788249247.npz"
    assert artifact["exists"] is True
