from pathlib import Path
from typing import Any, Dict, List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.dpnegf.results_unified import (
    AbacusOverlapResult,
    BuildSupercellResult,
    ConvertOverlapResult,
    PrepLammpsTasksResult,
    PrepNegfTasksResult,
    RunDpnegfLammpsResult,
    RunDpnegfResult,
    TransmissionPlotResult,
)
from dptb_pilot.tools.modules.dpnegf.submodules.lammps import (
    prepare_lammps_tasks,
    run_lammps_task,
)
from dptb_pilot.tools.modules.dpnegf.submodules.negf import (
    prepare_negf_tasks,
    run_negf_task,
)
from dptb_pilot.tools.modules.dpnegf.submodules.overlap import (
    convert_overlap,
    get_abacus_overlap,
)
from dptb_pilot.tools.modules.dpnegf.submodules.supercell import build_supercell
from dptb_pilot.tools.modules.dpnegf.submodules.transmission import _transmission_plot


@mcp.tool()
def dpnegf_build_supercell(
        init_conf_paths: List[Path],
        negf_config: Dict[str, Any],
        work_path: str = "."
) -> BuildSupercellResult:
    """Build stacked supercells for the DPNEGF workflow from VASP/POSCAR inputs."""
    return build_supercell(init_conf_paths=init_conf_paths, negf_config=negf_config, work_path=work_path)


@mcp.tool()
def dpnegf_prepare_lammps_tasks(
        stacked_system_paths: List[Path],
        system_infos: List[dict],
        relax_config: Dict[str, Any],
        inputs_config: Dict[str, Any],
        work_path: str = "."
) -> PrepLammpsTasksResult:
    """Prepare NEGF LAMMPS relaxation task directories."""
    return prepare_lammps_tasks(
        stacked_system_paths=stacked_system_paths,
        system_infos=system_infos,
        relax_config=relax_config,
        inputs_config=inputs_config,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_run_lammps_task(
        task_path: Path,
        task_name: str,
        deepmd_model_path: Path,
        relax_config: Dict[str, Any],
        work_path: str = "."
) -> RunDpnegfLammpsResult:
    """Run one prepared NEGF LAMMPS relaxation task."""
    return run_lammps_task(
        task_path=task_path,
        task_name=task_name,
        deepmd_model_path=deepmd_model_path,
        relax_config=relax_config,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_prepare_negf_tasks(
        negf_input_config: Dict[str, Any],
        task_infos: List[dict],
        task_config: Dict[str, Any],
        work_path: str = "."
) -> PrepNegfTasksResult:
    """Prepare per-task DPNEGF configs with device and lead atom ranges."""
    return prepare_negf_tasks(
        negf_input_config=negf_input_config,
        task_infos=task_infos,
        task_config=task_config,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_run_negf_task(
        modified_negf_input_config: Dict[str, Any],
        task_name: str,
        deeptb_model_path: Path,
        relaxed_system_archive_path: Path,
        negf_config: Dict[str, Any],
        work_path: str = "."
) -> RunDpnegfResult:
    """Run one DPNEGF task on a packed relaxed-system archive."""
    return run_negf_task(
        modified_negf_input_config=modified_negf_input_config,
        task_name=task_name,
        deeptb_model_path=deeptb_model_path,
        relaxed_system_archive_path=relaxed_system_archive_path,
        negf_config=negf_config,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_get_abacus_overlap(
        poscar_file_path: Path,
        input_file_path: Path,
        pp_file_paths: List[Path],
        orb_file_paths: List[Path],
        run_config: Dict[str, Any],
        work_path: str = "."
) -> AbacusOverlapResult:
    """Run ABACUS get_S for one relaxed POSCAR and collect sparse overlap output."""
    return get_abacus_overlap(
        poscar_file_path=poscar_file_path,
        input_file_path=input_file_path,
        pp_file_paths=pp_file_paths,
        orb_file_paths=orb_file_paths,
        run_config=run_config,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_convert_overlap(
        stru_path: Path,
        input_path: Path,
        running_log_path: Path,
        overlap_csr_path: Path,
        work_path: str = "."
) -> ConvertOverlapResult:
    """Convert ABACUS sparse overlap CSR output to dftio overlaps.h5."""
    return convert_overlap(
        stru_path=stru_path,
        input_path=input_path,
        running_log_path=running_log_path,
        overlap_csr_path=overlap_csr_path,
        work_path=work_path,
    )


@mcp.tool()
def dpnegf_transmission_plot(
        negf_out_path: Path,
        label: bool = False,
        work_path: str = ".",
        image_name: str = None,
) -> TransmissionPlotResult:
    """Plot transmission spectrum from a DPNEGF ``negf.out.pth`` file."""
    return _transmission_plot(
        negf_out_path=negf_out_path,
        label=label,
        work_path=work_path,
        image_name=image_name,
    )
