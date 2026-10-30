import jsonschema
import numpy as np
import pytest

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb import predict as predict_tools
from dptb_pilot.tools.modules.deeptb.submodules import band as band_module


def _write_band_structure(path, *, fermi_level=0.0):
    eigenvalues = np.array(
        [
            [-1.0, 0.4, 1.2],
            [-0.8, 0.6, 1.4],
        ]
    )
    np.savez(path, eigenvalues=eigenvalues, fermi_level=fermi_level)
    return eigenvalues


def test_band_gap_returns_edges_and_band_indices(tmp_path):
    band_path = tmp_path / "bands.npz"
    _write_band_structure(band_path)

    result = band_module._band_gap(band_path)

    assert result["band_gap"] == pytest.approx(1.2)
    assert result["gap_width"] == pytest.approx(1.2)
    assert result["vbm"] == result["gap_lower"] == pytest.approx(-0.8)
    assert result["cbm"] == result["gap_upper"] == pytest.approx(0.4)
    assert result["vbm_band_index"] == 0
    assert result["cbm_band_index"] == 1
    assert result["vbm_kpoint_index"] == 1
    assert result["cbm_kpoint_index"] == 0
    assert result["fermi_level"] == pytest.approx(0.0)
    assert result["fermi_level_source"] == "file:fermi_level"
    assert result["calculation_method"] == "fermi"
    assert result["status"] == "ok"
    assert result["is_valid"] is True
    assert result["diagnostic_message"] is None


def test_band_gap_uses_pseudo_fermi_level_for_smart_search(monkeypatch, tmp_path):
    band_path = tmp_path / "bands.npz"
    eigenvalues = _write_band_structure(band_path, fermi_level=9.0)
    captured = {}

    def fake_smart_band_gap(eig, fermi_level, atol, max_diff):
        captured.update(eig=eig, fermi_level=fermi_level, atol=atol, max_diff=max_diff)
        return {
            "gap_lower": -0.1,
            "gap_upper": 0.15,
            "gap_width": 0.25,
            "all_gaps": [(-0.1, 0.15)],
        }

    monkeypatch.setattr(band_module, "smart_band_gap", fake_smart_band_gap)

    result = band_module._band_gap(
        band_path,
        pseudo_fermi_level=0.0,
        smart_band_gap_atol=0.02,
    )

    assert result["band_gap"] == result["gap_width"] == pytest.approx(0.25)
    assert result["vbm"] == result["gap_lower"] == pytest.approx(-0.1)
    assert result["cbm"] == result["gap_upper"] == pytest.approx(0.15)
    assert result["fermi_level"] == pytest.approx(0.0)
    assert result["fermi_level_source"] == "argument:pseudo_fermi_level"
    assert result["calculation_method"] == "smart"
    assert result["is_valid"] is True
    assert result["all_gaps"] == [
        {"lower": -0.1, "upper": 0.15, "width": pytest.approx(0.25)}
    ]
    np.testing.assert_array_equal(captured["eig"], eigenvalues)
    assert captured["fermi_level"] == pytest.approx(0.0)
    assert captured["atol"] == pytest.approx(0.02)
    assert captured["max_diff"] == pytest.approx(1.0)


def test_band_gap_rejects_two_fermi_levels(tmp_path):
    band_path = tmp_path / "bands.npz"
    _write_band_structure(band_path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        band_module._band_gap(
            band_path,
            fermi_level=0.0,
            pseudo_fermi_level=0.0,
        )


@pytest.mark.parametrize("extension", [".npy", ".npz"])
def test_band_gap_normalizes_mapping_file_formats(tmp_path, extension):
    path = tmp_path / f"bands{extension}"
    eigenvalues = np.array([[-1.0, 0.5], [-0.9, 0.7]])
    payload = {"eigenvalues": eigenvalues, "E_fermi": np.array(0.0)}
    if extension == ".npy":
        np.save(path, payload)
    else:
        np.savez(path, **payload)

    result = band_module._band_gap(path)

    assert result["band_gap"] == pytest.approx(1.4)
    assert result["fermi_level_source"] == "file:E_fermi"


def test_band_gap_accepts_numeric_npy_and_single_channel_shape(tmp_path):
    path = tmp_path / "bands.npy"
    np.save(path, np.array([[[-1.0, 0.5], [-0.8, 0.6]]]))

    result = band_module._band_gap(path, fermi_level=0.0)

    assert result["band_gap"] == pytest.approx(1.3)
    assert result["fermi_level_source"] == "argument:fermi_level"


def test_band_gap_uses_xlist_to_normalize_transposed_eigenvalues(tmp_path):
    path = tmp_path / "bands.npz"
    np.savez(
        path,
        eigenvalues=np.array([[-1.0, -0.8], [0.4, 0.6], [1.2, 1.4]]),
        xlist=np.array([0.0, 1.0]),
        fermi_level=0.0,
    )

    _, eigenvalues = band_module._load_band_gap_input(path)
    result = band_module._band_gap(path)

    assert eigenvalues.shape == (2, 3)
    assert result["vbm_band_index"] == 0
    assert result["cbm_band_index"] == 1


def test_band_gap_rejects_inconsistent_xlist(tmp_path):
    path = tmp_path / "bands.npz"
    np.savez(path, eigenvalues=np.zeros((2, 3)), xlist=np.arange(4))

    with pytest.raises(ValueError, match="xlist.*does not match"):
        band_module._band_gap(path, fermi_level=0.0)


def test_band_gap_reports_fermi_level_outside_spectrum(tmp_path):
    path = tmp_path / "bands.npy"
    np.save(path, np.array([[1.0, 2.0], [1.5, 2.5]]))

    result = band_module._band_gap(path, fermi_level=0.0)

    assert result["band_gap"] == result["gap_width"] == 0.0
    assert result["vbm"] is None
    assert result["cbm"] is None
    assert result["status"] == "fermi_level_outside_spectrum"
    assert result["is_valid"] is False
    assert result["risky"] is True


def test_smart_band_gap_handles_too_few_candidates_without_index_error(tmp_path):
    path = tmp_path / "bands.npy"
    np.save(path, np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))

    result = band_module._band_gap(
        path,
        pseudo_fermi_level=0.0,
        smart_band_gap_atol=1.0,
    )

    assert result["band_gap"] == 0.0
    assert result["status"] == "insufficient_fermi_bracketing"
    assert result["is_valid"] is False
    assert result["diagnostic_message"]


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.empty((0, 2)), "must not be empty"),
        (np.zeros((2, 2, 2)), "must have shape"),
        (np.array([[0.0, np.nan]]), "NaN or infinite"),
    ],
)
def test_band_gap_rejects_invalid_eigenvalues(tmp_path, eigenvalues, message):
    path = tmp_path / "bands.npy"
    np.save(path, eigenvalues)

    with pytest.raises(ValueError, match=message):
        band_module._band_gap(path, fermi_level=0.0)


@pytest.mark.parametrize("n_atoms", [0, -1, 1.5, True])
def test_band_gap_rejects_invalid_atom_count(tmp_path, n_atoms):
    path = tmp_path / "bands.npy"
    np.save(path, np.array([[-1.0, 1.0]]))

    with pytest.raises(ValueError, match="positive integer"):
        band_module._band_gap(path, n_atoms=n_atoms)


def test_band_gap_mcp_output_schema_accepts_unified_result(tmp_path):
    assert predict_tools.band_gap is not None
    tool = mcp.mcp._tool_manager._tools["band_gap"]
    band_path = tmp_path / "bands.npz"
    _write_band_structure(band_path)

    result = band_module._band_gap(band_path)
    _, structured_content = tool.fn_metadata.convert_result(result)

    jsonschema.validate(structured_content, tool.fn_metadata.output_schema)
