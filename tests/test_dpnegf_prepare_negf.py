from dptb_pilot.tools.modules.dpnegf.submodules.negf import prepare_negf_tasks


def test_dpnegf_prepare_negf_tasks_rewrites_ranges(tmp_path):
    negf_input_config = {
        "task_options": {
            "stru_options": {
                "device": {"id": "old-device"},
                "lead_L": {"id": "old-left"},
                "lead_R": {"id": "old-right"},
            }
        },
        "structure": "old.vasp",
    }
    task_infos = [{
        "conf_name": "stacked_10_0",
        "temp": 300,
        "pres": 0,
        "system_info": {"atom_index": [20, 80, 100]},
    }]

    result = prepare_negf_tasks(negf_input_config, task_infos, {"use_external_overlap": False}, str(tmp_path))

    assert result["task_names"] == ["negf_stacked_10_0_300K_0bar"]
    modified = result["modified_negf_input_configs"][0]
    assert modified["structure"] == "relaxed.vasp"
    assert modified["task_options"]["stru_options"]["device"]["id"] == "20-80"
    assert modified["task_options"]["stru_options"]["lead_L"]["id"] == "0-20"
    assert modified["task_options"]["stru_options"]["lead_R"]["id"] == "80-100"
    assert negf_input_config["structure"] == "old.vasp"
