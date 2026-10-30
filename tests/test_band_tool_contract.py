from pathlib import Path

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb import predict


def test_band_predict_exposes_and_forwards_optional_parameters(monkeypatch):
    captured = {}

    def fake_band_predict(**kwargs):
        captured.update(kwargs)
        return {
            "band_structure_file_path": Path("bands.npz"),
            "image_file_path": Path("bands.png"),
            "fermi_level": 0.0,
        }

    monkeypatch.setattr(predict, "_band_predict", fake_band_predict)
    overlap = Path("overlap.h5")
    predict.band_predict(
        model_file_path=Path("model.pth"),
        structure_file_path=Path("POSCAR"),
        kpath="[]",
        nel_atom={"Si": 4},
        override_overlap=overlap,
        smearing_method="Gaussian",
        sigma_val=0.05,
    )

    assert captured["override_overlap"] == overlap
    assert captured["smearing_method"] == "Gaussian"
    assert captured["sigma_val"] == 0.05
    schema = mcp.mcp._tool_manager._tools["band_predict"].parameters
    assert {"override_overlap", "smearing_method", "sigma_val"} <= set(
        schema["properties"]
    )


def test_band_compare_exposes_and_forwards_alignment_parameters(monkeypatch):
    captured = {}

    def fake_band_compare(**kwargs):
        captured.update(kwargs)
        return {"band_compare_path": Path("comparison.png")}

    monkeypatch.setattr(predict, "_band_compare", fake_band_compare)
    predict.band_compare(
        dptb_result_path=Path("bands.npz"),
        dft_result_path=Path("OUT.ABACUS"),
        match_fermi_level=True,
        delta=0.25,
    )

    assert captured["match_fermi_level"] is True
    assert captured["delta"] == 0.25
    schema = mcp.mcp._tool_manager._tools["band_compare"].parameters
    assert {"match_fermi_level", "delta"} <= set(schema["properties"])
