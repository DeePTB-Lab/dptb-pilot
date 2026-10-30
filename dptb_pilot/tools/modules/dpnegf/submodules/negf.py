import copy
import os
from pathlib import Path
from typing import Any, Dict, List

from dptb_pilot.tools.modules.dpnegf.submodules.archive import pack_files, unpack_files
from dptb_pilot.tools.modules.util.comm import temporary_chdir


def prepare_negf_tasks(
        negf_input_config: Dict[str, Any],
        task_infos: List[Dict[str, Any]],
        task_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Prepare per-relaxed-structure DPNEGF input configs."""
    Path(work_path).absolute().mkdir(parents=True, exist_ok=True)
    task_names: List[str] = []
    modified_configs: List[Dict[str, Any]] = []

    for task_info in task_infos:
        system_info = task_info["system_info"]
        modified_config = copy.deepcopy(negf_input_config)
        stru_options = modified_config["task_options"]["stru_options"]
        stru_options["device"]["id"] = f"{system_info['atom_index'][0]}-{system_info['atom_index'][1]}"
        stru_options["lead_L"]["id"] = f"0-{system_info['atom_index'][0]}"
        stru_options["lead_R"]["id"] = f"{system_info['atom_index'][1]}-{system_info['atom_index'][2]}"
        modified_config["structure"] = "relaxed.vasp"

        task_name = f"negf_{task_info['conf_name']}_{task_info['temp']}K_{task_info['pres']}bar"
        task_names.append(task_name)
        modified_configs.append(modified_config)

    return {"task_names": task_names, "modified_negf_input_configs": modified_configs}


def _run_single_negf(
        sys_path: Path,
        deeptb_model_path: Path,
        modified_negf_input_config: Dict[str, Any],
        use_common_self_energy: bool,
) -> None:
    import logging

    import matplotlib.pyplot as plt
    import torch
    from dpnegf.negf.lead_property import _has_saved_self_energy
    from dpnegf.runner.NEGF import NEGF
    from dpnegf.utils.loggers import set_log_handles
    from dptb.nn.build import build_model

    run_dir = Path(sys_path.name.replace(".", "_"))
    run_dir.mkdir(parents=True, exist_ok=True)
    with temporary_chdir(run_dir):
        log_path = Path("log")
        set_log_handles(logging.INFO, log_path)

        model_target = Path(deeptb_model_path.name)
        if not model_target.exists():
            os.symlink(deeptb_model_path, model_target)
        model = build_model(model_target.name, common_options={"device": "cpu"})

        relaxed_target = Path("relaxed.vasp")
        if not relaxed_target.exists():
            os.symlink(sys_path, relaxed_target)

        atomic_data_options = modified_negf_input_config.get("AtomicData_options")
        self_energy_save_path = (Path("..") / "common_self_energy").absolute() if use_common_self_energy else Path(".").absolute()
        negf = NEGF(
            model=model,
            AtomicData_options=atomic_data_options,
            structure="relaxed.vasp",
            results_path=".",
            self_energy_save_path=self_energy_save_path,
            use_saved_se=_has_saved_self_energy(self_energy_save_path),
            **modified_negf_input_config["task_options"],
        )
        negf.compute()

        negf_out = torch.load("negf.out.pth")
        plt.plot(negf_out["uni_grid"], negf_out["DOS"][str(negf_out["k"][0])])
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.title("DOS vs Energy")
        plt.grid()
        plt.savefig("dos.png")
        plt.close()

        plt.plot(negf_out["uni_grid"], negf_out["T_avg"])
        plt.xlabel("Energy (eV)")
        plt.ylabel("Transmission")
        plt.title("Transmission vs Energy")
        plt.grid()
        plt.savefig("transmission.png")
        plt.close()


def run_negf_task(
        modified_negf_input_config: Dict[str, Any],
        task_name: str,
        deeptb_model_path: Path,
        relaxed_system_archive_path: Path,
        negf_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Run DPNEGF for all relaxed systems in one relaxed-system archive."""
    deeptb_model_path = Path(deeptb_model_path).absolute()
    work_dir = Path(work_path).absolute() / task_name
    work_dir.mkdir(parents=True, exist_ok=True)
    relaxed_systems = unpack_files(relaxed_system_archive_path, work_dir)
    use_common_self_energy = negf_config.get("use_common_self_energy", True)

    log_paths: List[Path] = []
    extra_output_archives: List[Path] = []
    negf_result_paths: List[Path] = []

    with temporary_chdir(work_dir):
        for sys_path in relaxed_systems:
            _run_single_negf(sys_path, deeptb_model_path, modified_negf_input_config, use_common_self_energy)
            run_dir = Path(sys_path.name.replace(".", "_"))
            log_paths.append((work_dir / run_dir / "log").absolute())
            extra_names = [name for name in ["dos.png", "transmission.png", "profile_report.html"] if (run_dir / name).exists()]
            extra_output_archives.append(pack_files(run_dir, extra_names, "extra_outputs.tar.gz"))
            negf_result_paths.append((work_dir / run_dir / "negf.out.pth").absolute())

    return {
        "log_paths": log_paths,
        "negf_result_paths": negf_result_paths,
        "extra_output_archives": extra_output_archives,
        "task_name": task_name,
    }
