from pathlib import Path

from dptb_pilot.tools.modules.dpnegf.submodules.supercell import build_supercell


RESOURCE = Path("negfflow/example/api/init_confs/10_0.vasp")


def test_dpnegf_build_supercell(tmp_path: Path):
    negf_config = {
        "supercell": {"lead_L": 2, "device": 2, "lead_R": 2},
        "direction": "z",
    }

    result = build_supercell([RESOURCE], negf_config, work_path=str(tmp_path))

    assert len(result["stacked_system_paths"]) == 1
    assert result["stacked_system_paths"][0].exists()
    assert result["stacked_system_paths"][0].name == "stacked_10_0.vasp"
    assert len(result["system_infos"]) == 1
    info = result["system_infos"][0]
    assert info["atom_number"] > 0
    assert info["atom_index"] == [2 * info["atom_number"], 4 * info["atom_number"], 6 * info["atom_number"]]
