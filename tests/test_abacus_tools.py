import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dptb_pilot.tools.modules.abacus import core


def test_prepare_run_directory_updates_stru_pp_and_orb_filenames(tmp_path):
    source = tmp_path / "source"
    run = tmp_path / "run"
    source.mkdir()
    run.mkdir()
    stru = source / "STRU.source"
    stru.write_text(
        """ATOMIC_SPECIES
C 12.011 old/legacy-C.upf upf201 # keep pp metadata
Si 28.085 C:\\legacy\\legacy-Si.upf

NUMERICAL_ORBITAL
old/legacy-C.orb # carbon orbital
legacy-Si.orb

LATTICE_CONSTANT
1.0
""",
        encoding="utf-8",
    )
    paths = [
        source / "Si-new.UPF",
        source / "C-new.orb",
        source / "C-new.upf",
        source / "Si-new.orb",
    ]
    for path in paths:
        path.write_text(path.name, encoding="utf-8")

    core._prepare_run_directory(run, stru, paths, None)

    prepared = (run / "STRU").read_text(encoding="utf-8")
    assert "C 12.011 C-new.upf upf201 # keep pp metadata" in prepared
    assert "Si 28.085 Si-new.UPF" in prepared
    assert "C-new.orb # carbon orbital" in prepared
    assert "Si-new.orb" in prepared
    assert "LATTICE_CONSTANT\n1.0" in prepared
    assert all(
        (run / path.name).read_text(encoding="utf-8") == path.name for path in paths
    )


def test_prepare_run_directory_prefers_exact_names_and_allows_partial_lists(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    stru = tmp_path / "STRU.source"
    stru.write_text(
        """ATOMIC_SPECIES
C 12 old-C.upf
Si 28 Si.upf

NUMERICAL_ORBITAL
C.orb
Si.orb
""",
        encoding="utf-8",
    )
    si_pp = tmp_path / "Si.upf"
    c_pp = tmp_path / "renamed-C.upf"
    si_pp.write_text("si", encoding="utf-8")
    c_pp.write_text("c", encoding="utf-8")

    core._prepare_run_directory(run, stru, [si_pp, c_pp], None)

    prepared = (run / "STRU").read_text(encoding="utf-8")
    assert "C 12 renamed-C.upf" in prepared
    assert "Si 28 Si.upf" in prepared
    assert "C.orb\nSi.orb" in prepared


@pytest.mark.parametrize("with_kpt", [False, True])
def test_run_abacus_returns_complete_work_directory_for_dftio(
    monkeypatch, tmp_path, with_kpt
):
    from dptb_pilot.tools.modules.deeptb.submodules import dftio

    source = tmp_path / "source"
    source.mkdir()
    inputs = {
        "STRU": "ATOMIC_SPECIES\nSi 28 Si.upf\n\nNUMERICAL_ORBITAL\nSi.orb\n",
        "INPUT": "INPUT_PARAMETERS\ncalculation scf\n",
        "Si.upf": "pseudopotential",
        "Si.orb": "orbital",
    }
    if with_kpt:
        inputs["KPT"] = "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"
    for name, content in inputs.items():
        (source / name).write_text(content, encoding="utf-8")
    extra = source / "extra"
    extra.mkdir()
    (extra / "data").write_text("supporting data", encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(core, "generate_work_path", lambda: str(work))
    run_paths = []

    def fake_abacus(command, cwd, **kwargs):
        run_path = Path(cwd)
        run_paths.append(run_path)
        output = run_path / "OUT.ABACUS"
        output.mkdir()
        (output / "running_scf.log").write_text("EFERMI = 1.0\n", encoding="utf-8")
        (run_path / "execution.log").write_text("finished", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(core.sp, "run", fake_abacus)
    result = core._run_abacus(
        source / "STRU", source / "INPUT",
        kpt_path=source / "KPT" if with_kpt else None,
        pp_orb_paths=[source / "Si.upf", source / "Si.orb"],
        other_file_path=extra,
    )

    root = result["abacus_output_path"]
    assert isinstance(root, Path)
    assert root == work
    assert not run_paths[0].exists()
    for name, content in inputs.items():
        assert (root / name).read_text(encoding="utf-8") == content
    assert (root / "KPT").exists() == with_kpt
    assert (root / "extra" / "data").read_text(encoding="utf-8") == "supporting data"
    assert (root / "execution.log").read_text(encoding="utf-8") == "finished"
    assert core._abacus_get_fermi_level(root) == {"fermi_level": 1.0}

    parse_work = tmp_path / "parse_work"
    parse_work.mkdir()
    monkeypatch.setattr(dftio, "generate_work_path", lambda: str(parse_work))
    monkeypatch.setattr(dftio, "_dftio_executable", lambda: ["dftio"])

    def fake_dftio(command, cwd, **kwargs):
        normalized_root = Path(command[command.index("-r") + 1])
        staged, = normalized_root.iterdir()
        assert (staged / "INPUT").read_text(encoding="utf-8") == inputs["INPUT"]
        assert (staged / "STRU").read_text(encoding="utf-8") == inputs["STRU"]
        assert (staged / "Si.upf").is_file()
        assert (staged / "Si.orb").is_file()
        assert (staged / "OUT.ABACUS" / "running_scf.log").is_file()
        assert not (staged / run_paths[0].name).exists()
        parsed = Path(command[command.index("-o") + 1])
        (parsed / "Si.0").mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(dftio.sp, "run", fake_dftio)
    parsed = dftio._dftio_parse(root)
    assert Path(parsed["input_mapping"][str(root.resolve())]).is_dir()


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
    root = result["abacus_output_path"]
    assert isinstance(result["abacus_output_path"], Path)
    assert root == work_path
    assert (root / "STRU").read_text(encoding="utf-8") == "structure"
    assert (root / "INPUT").read_text(encoding="utf-8") == nscf_input.read_text(encoding="utf-8")
    assert (root / "KPT").read_text(encoding="utf-8") == kpt.read_text(encoding="utf-8")
    output = root / "OUT.ABACUS"
    assert (output / "charge.cube").read_text(encoding="utf-8") == "scf charge"
    assert (output / "running_nscf.log").exists()


@pytest.mark.parametrize("use_work_directory", [False, True])
def test_abacus_band_gap_reads_text_band_format(tmp_path, use_work_directory):
    output = tmp_path / "OUT.ABACUS"
    output.mkdir()
    (output / "running_nscf.log").write_text("EFERMI = 0.5 eV\n", encoding="utf-8")
    np.savetxt(
        output / "BANDS_1.dat",
        np.array([[1, 0.0, 0.2, 0.9], [2, 1.0, 0.3, 1.1]]),
    )

    result = core._abacus_band_gap(tmp_path if use_work_directory else output)

    assert result["vbm"] == pytest.approx(-0.2)
    assert result["cbm"] == pytest.approx(0.4)
    assert result["band_gap"] == pytest.approx(0.6)


@pytest.mark.parametrize("use_work_directory", [False, True])
def test_abacus_band_plot_accepts_work_or_output_directory(
    monkeypatch, tmp_path, use_work_directory
):
    root = tmp_path / "calculation"
    output = root / "OUT.ABACUS"
    output.mkdir(parents=True)
    (output / "running_scf.log").write_text("EFERMI = 0.5\n", encoding="utf-8")
    np.savetxt(output / "BANDS_1.dat", [[1, 0.0, 0.2, 0.9], [2, 1.0, 0.3, 1.1]])
    monkeypatch.setattr(core, "generate_work_path", lambda: str(tmp_path))

    result = core._abacus_band_plot(root if use_work_directory else output)

    assert result["abacus_output_path"].is_file()
    assert result["abacus_output_path"].suffix == ".png"


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

    def fake_smart_band_gap(eig, fermi_level, atol, max_diff):
        captured.update(eig=eig, fermi_level=fermi_level, atol=atol, max_diff=max_diff)
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
    assert captured["fermi_level"] == 0.5
    assert captured["atol"] == 0.02
    assert captured["max_diff"] == 2.0
    assert result["band_gap"] == pytest.approx(0.7)
    assert result["vbm"] == pytest.approx(-0.3)
    assert result["cbm"] == pytest.approx(0.4)
