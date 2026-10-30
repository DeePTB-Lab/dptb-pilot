from pathlib import Path

from dptb_pilot.tools.modules.dpnegf.submodules.lammps import prepare_lammps_tasks
from dptb_pilot.tools.modules.dpnegf.submodules.supercell import build_supercell


RESOURCE = Path("negfflow/example/api/init_confs/10_0.vasp")


def test_dpnegf_prepare_lammps_tasks(tmp_path: Path):
    supercell_result = build_supercell(
        [RESOURCE],
        {"supercell": {"lead_L": 2, "device": 2, "lead_R": 2}, "direction": "z"},
        work_path=str(tmp_path / "supercell"),
    )
    relax_config = {
        "ensemble": "nvt",
        "device_end_fixed_radius": 1.0,
        "dt": 0.001,
        "nsteps": 10,
        "temps": [300],
        "press": [0],
        "run_config": {"command": "lmp"},
    }
    inputs_config = {"deepmd_model_path": "frozen_model.pb", "deepmd_model_type_map": ["C"]}

    result = prepare_lammps_tasks(
        supercell_result["stacked_system_paths"],
        supercell_result["system_infos"],
        relax_config,
        inputs_config,
        work_path=str(tmp_path / "lammps"),
    )

    assert len(result["task_paths"]) == 1
    task_path = result["task_paths"][0]
    assert task_path.exists()
    assert (task_path / "lammps.data").exists()
    in_lammps = (task_path / "in.lammps").read_text(encoding="utf-8")
    assert "pair_style      deepmd frozen_model.pb" in in_lammps
    assert "fix             1 mobile nvt temp 300 300 0.1" in in_lammps
    assert result["task_infos"][0]["system_info"] == supercell_result["system_infos"][0]
