import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dptb_pilot.tools.modules.abacus import core


def test_run_abacus_scf_nscf_switches_input_and_kpt(monkeypatch, tmp_path):
    stru = tmp_path / "STRU.source"
    scf_input = tmp_path / "INPUT_scf"
    nscf_input = tmp_path / "INPUT_nscf"
    kpt = tmp_path / "KPT.source"
    stru.write_text("structure", encoding="utf-8")
    scf_input.write_text("calculation scf\nkspacing 0.08", encoding="utf-8")
    nscf_input.write_text(
        "calculation nscf\nread_file_dir ./OUT.ABACUS/\ninit_chg file",
        encoding="utf-8",
    )
    kpt.write_text("K_POINTS\n2\nLine", encoding="utf-8")

    work_path = tmp_path / "work"
    work_path.mkdir()
    monkeypatch.setattr(core, "generate_work_path", lambda: str(work_path))
    stages = []

    def fake_run(command, cwd, capture_output, text):
        run_path = Path(cwd)
        stages.append(
            {
                "command": command,
                "input": (run_path / "INPUT").read_text(encoding="utf-8"),
                "kpt": (run_path / "KPT").read_text(encoding="utf-8")
                if (run_path / "KPT").exists()
                else None,
                "charge_exists": (run_path / "OUT.ABACUS" / "charge.cube").exists(),
            }
        )
        output = run_path / "OUT.ABACUS"
        output.mkdir(exist_ok=True)
        if len(stages) == 1:
            (output / "charge.cube").write_text("scf charge", encoding="utf-8")
            (output / "running_scf.log").write_text("EFERMI = 1.0", encoding="utf-8")
        else:
            (output / "running_nscf.log").write_text("EFERMI = 1.1", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(core.sp, "run", fake_run)

    result = core._run_abacus_scf_nscf(
        stru,
        scf_input,
        nscf_input,
        kpt,
        command="mpirun -np 4 abacus",
    )

    assert stages[0]["command"] == ["mpirun", "-np", "4", "abacus"]
    assert "calculation scf" in stages[0]["input"]
    assert stages[0]["kpt"] is None
    assert stages[1]["charge_exists"] is True
    assert "calculation nscf" in stages[1]["input"]
    assert stages[1]["kpt"] == "K_POINTS\n2\nLine"
    output = Path(result["abacus_output_path"])
    assert (output / "charge.cube").read_text(encoding="utf-8") == "scf charge"
    assert (output / "running_nscf.log").exists()


def test_abacus_band_gap_reads_text_band_format(tmp_path):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (output / "running_nscf.log").write_text("EFERMI = 0.5 eV\n", encoding="utf-8")
    np.savetxt(
        output / "BANDS_1.dat",
        np.array([[1, 0.0, 0.2, 0.9], [2, 1.0, 0.3, 1.1]]),
    )

    result = core._abacus_band_gap(output)

    assert result["vbm"] == pytest.approx(-0.2)
    assert result["cbm"] == pytest.approx(0.4)
    assert result["band_gap"] == pytest.approx(0.6)


def test_abacus_smart_band_gap_adapts_text_bands_for_deeptb(monkeypatch, tmp_path):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (output / "running_nscf.log").write_text("EFERMI = 0.5\n", encoding="utf-8")
    np.savetxt(
        output / "BANDS_1.dat",
        np.array([[1, 0.0, 0.1, 0.9], [2, 1.0, 0.2, 1.0]]),
    )

    captured = {}
    fake_band_module = types.ModuleType(
        "dptb_pilot.tools.modules.deeptb.submodules.band"
    )

    def fake_smart_band_gap(eig, efermi, atol, max_diff):
        captured.update(eig=eig, efermi=efermi, atol=atol, max_diff=max_diff)
        return {"gap_lower": 0.2, "gap_upper": 0.9, "gap_width": 0.7}

    fake_band_module.smart_band_gap = fake_smart_band_gap
    monkeypatch.setitem(
        sys.modules,
        "dptb_pilot.tools.modules.deeptb.submodules.band",
        fake_band_module,
    )

    result = core._abacus_band_gap(
        output,
        use_smart_band_gap=True,
        smart_band_gap_atol=0.02,
        smart_band_gap_max_diff=2.0,
    )

    np.testing.assert_allclose(captured["eig"], [[0.1, 0.9], [0.2, 1.0]])
    assert captured["efermi"] == 0.5
    assert captured["atol"] == 0.02
    assert captured["max_diff"] == 2.0
    assert result["band_gap"] == pytest.approx(0.7)
    assert result["vbm"] == pytest.approx(-0.3)
    assert result["cbm"] == pytest.approx(0.4)
