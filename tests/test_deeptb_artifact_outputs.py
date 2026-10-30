from pathlib import Path
import asyncio

import numpy as np

from dptb_pilot.tools.modules.deeptb.submodules import band as band_module


class FakeBandData:
    def export(self, output_path):
        np.savez(str(output_path) + ".npz", eigenvalues=np.zeros((2, 3)))

    def plot(self, filename, **_kwargs):
        Path(filename).write_bytes(b"png")


class FakeBandAccessor:
    def set_kpath(self, **_kwargs):
        pass

    def compute(self):
        return FakeBandData()


class FakeTBSystem:
    def __init__(self, **_kwargs):
        self.band = FakeBandAccessor()

    def set_electrons(self, **_kwargs):
        pass

    def get_efermi(self, **_kwargs):
        return 1.25


def test_band_predict_returns_uploadable_path_artifacts(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    model = tmp_path / "model.pth"
    structure = tmp_path / "POSCAR"
    model.write_bytes(b"model")
    structure.write_text("structure", encoding="utf-8")

    monkeypatch.setattr(band_module, "TBSystem", FakeTBSystem)
    monkeypatch.setattr(band_module, "generate_work_path", lambda: str(scratch))

    result = band_module._band_predict(
        model_file_path=model,
        structure_file_path=structure,
        kpath='[[0.0, 0.0, 0.0, 1, "G"]]',
        nel_atom={"Si": 4},
        work_path=str(tmp_path / "outputs"),
    )

    assert isinstance(result["band_structure_file_path"], Path)
    assert result["band_structure_file_path"].suffix == ".npz"
    assert result["band_structure_file_path"].is_file()
    assert isinstance(result["image_file_path"], Path)
    assert result["image_file_path"].is_file()


def test_calculation_server_uploads_band_npz(monkeypatch, tmp_path):
    from dp.agent.server import calculation_mcp_server

    band_file = tmp_path / "bandstructure.npz"
    image_file = tmp_path / "band.png"
    band_file.write_bytes(b"npz")
    image_file.write_bytes(b"png")

    class FakeStorage:
        def __init__(self):
            self.uploaded = []

        def upload(self, key, path):
            self.uploaded.append((key, Path(path)))
            return key

    storage = FakeStorage()
    monkeypatch.setattr(
        calculation_mcp_server,
        "init_storage",
        lambda _config: ("local", storage),
    )

    results, artifacts = calculation_mcp_server.handle_output_artifacts(
        {
            "band_structure_file_path": band_file,
            "image_file_path": image_file,
            "fermi_level": 1.25,
        },
        "exec-1",
        None,
    )

    assert {path for _, path in storage.uploaded} == {band_file, image_file}
    assert results["band_structure_file_path"].startswith("local://")
    assert "band_structure_file_path" in artifacts


def test_server_automatically_stages_tool_output_uri(monkeypatch, tmp_path):
    from dptb_pilot.server import app as server_app

    session_id = "S" * 32

    class FakeDownloadStorage:
        def download(self, key, destination):
            target = Path(destination) / Path(key).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"npz")
            return str(target)

    monkeypatch.setattr(server_app, "work_path", str(tmp_path))
    monkeypatch.setattr(
        server_app.calculation_mcp_server,
        "init_storage",
        lambda _config: ("local", FakeDownloadStorage()),
    )

    staged = asyncio.run(server_app._stage_output_artifact_uris(
        session_id,
        [{"band_structure_file_path": "local://exec/outputs/bandstructure.npz"}],
    ))

    expected = tmp_path / session_id / "files" / "bandstructure.npz"
    assert staged == [str(expected.resolve())]
    assert expected.read_bytes() == b"npz"
