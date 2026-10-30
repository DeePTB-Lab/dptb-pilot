from pathlib import Path
import shutil
import subprocess

import pytest
from ase.io import read

from dptb_pilot.tools.modules.lammps.conversion import convert_structure_to_lammps_data
from dptb_pilot.tools.modules.lammps.execution import execute_lammps
from dptb_pilot.tools.modules.lammps.inputs import (
    generate_lammps_input_file,
    generate_lammps_inputs,
)


STRUCTURE = Path(__file__).parent / "resources" / "10_0.vasp"


def test_generate_lammps_inputs_for_multiple_methods(tmp_path):
    result = generate_lammps_inputs(
        structure_file_path=STRUCTURE,
        output_path=tmp_path / "jobs",
        methods=["single", "relax", "npt"],
        strains=[0.0, 1.0],
        axis="z",
        target_length=2.0,
        model_name="/models/potential.pb",
        temps=[300],
        press=[1],
        nsteps=20,
    )

    assert set(result["method_results"]) == {"single", "relax", "npt"}
    for method, method_result in result["method_results"].items():
        assert len(method_result["task_paths"]) == 2
        assert all((path / "lammps.data").is_file() for path in method_result["task_paths"])
        assert all((path / "in.lammps").is_file() for path in method_result["task_paths"])
        script = method_result["in_lammps_file_paths"][0].read_text(encoding="utf-8")
        assert "pair_style      deepmd potential.pb" in script
        if method == "single":
            assert "run             0" in script
        elif method == "relax":
            assert "minimize" in script
        else:
            assert "fix             1 all npt temp 300.0 300.0 0.1 iso 1.000000 1.000000" in script


def test_legacy_single_generator_parameter_name_is_preserved(tmp_path):
    result = generate_lammps_input_file(
        struture_file_path=STRUCTURE,
        output_path=tmp_path / "legacy",
        lammps_run_method="single",
        target_length=2.0,
    )

    assert result["in_lammps_file_paths"][0].is_file()
    assert result["lammps_data_file_paths"][0].is_file()


def test_convert_structure_to_lammps_data_preserves_explicit_type_order(tmp_path):
    output_path = tmp_path / "converted" / "lammps.data"
    result = convert_structure_to_lammps_data(
        STRUCTURE,
        output_file_path=output_path,
        element_order=["C"],
    )

    assert result["lammps_data_file_path"] == output_path.resolve()
    assert result["element_order"] == ["C"]
    converted = read(output_path, format="lammps-data", Z_of_type={1: 6})
    original = read(STRUCTURE)
    assert converted.get_chemical_symbols() == original.get_chemical_symbols()


def test_execute_lammps_stages_files_and_collects_relaxed_structure(tmp_path, monkeypatch):
    input_file = tmp_path / "in.lammps"
    data_file = tmp_path / "lammps.data"
    input_file.write_text("read_data lammps.data\n", encoding="utf-8")
    from ase.io import write

    write(data_file, read(STRUCTURE), format="lammps-data")

    def fake_run(command, cwd, capture_output, text, check):
        run_dir = Path(cwd)
        shutil.copy2(run_dir / "lammps.data", run_dir / "relaxed.data")
        (run_dir / "log.lammps").write_text("LAMMPS done", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr("dptb_pilot.tools.modules.lammps.execution.subprocess.run", fake_run)
    result = execute_lammps(
        in_lammps_file_path=input_file,
        lammps_data_file_path=data_file,
        lmp_command="lmp -sf omp",
        output_path=tmp_path / "runs",
        task_name="sample",
    )

    assert result["task_path"] == (tmp_path / "runs" / "sample").resolve()
    assert result["log_file_path"].is_file()
    assert result["relaxed_data_file_path"].is_file()
    assert result["relaxed_structure_file_path"].is_file()
    assert result["command"].startswith("lmp -sf omp -i in.lammps")
    with pytest.raises(FileExistsError):
        execute_lammps(
            in_lammps_file_path=input_file,
            lammps_data_file_path=data_file,
            lmp_command="lmp",
            output_path=tmp_path / "runs",
            task_name="sample",
        )
