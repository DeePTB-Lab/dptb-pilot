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
    """Build stacked supercells for the DPNEGF workflow from VASP/POSCAR inputs.

    Parameters
    ----------
    init_conf_paths : List[Path]
        Input VASP/POSCAR structures.
    negf_config : Dict[str, Any]
        Stacking configuration containing ``direction`` and the ``supercell``
        lead/device layer counts; may also contain ``auto_scale_length``.
    work_path : str, optional
        Directory where stacked structures are written.

    Returns
    -------
    BuildSupercellResult
        Stacked structure paths and atom-range metadata for later workflow stages.
    """
    return build_supercell(init_conf_paths=init_conf_paths, negf_config=negf_config, work_path=work_path)


@mcp.tool()
def dpnegf_prepare_lammps_tasks(
        stacked_system_paths: List[Path],
        system_infos: List[dict],
        relax_config: Dict[str, Any],
        inputs_config: Dict[str, Any],
        work_path: str = "."
) -> PrepLammpsTasksResult:
    """Prepare NEGF LAMMPS relaxation task directories.

    Parameters
    ----------
    stacked_system_paths : List[Path]
        Structures produced by ``dpnegf_build_supercell``.
    system_infos : List[dict]
        Atom counts and lead/device atom boundaries paired with the structures.
    relax_config : Dict[str, Any]
        Temperatures, pressures, ensemble settings, timestep, step count, and run config.
    inputs_config : Dict[str, Any]
        DeepMD model path and optional element type map.
    work_path : str, optional
        Parent directory for generated task directories.

    Returns
    -------
    PrepLammpsTasksResult
        Task root, paths, names, and per-task metadata.
    """
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
    """Run one prepared NEGF LAMMPS relaxation task.

    Parameters
    ----------
    task_path : Path
        Prepared task directory containing ``in.lammps`` and ``lammps.data``.
    task_name : str
        Name used for the run directory and returned task identifier.
    deepmd_model_path : Path
        DeepMD model copied into the task directory.
    relax_config : Dict[str, Any]
        Ensemble and command settings used to run and collect the calculation.
    work_path : str, optional
        Parent directory for the run output.

    Returns
    -------
    RunDpnegfLammpsResult
        Log path and archive containing the relaxed structures.
    """
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
    """Prepare per-task DPNEGF configs with device and lead atom ranges.

    Parameters
    ----------
    negf_input_config : Dict[str, Any]
        Base DPNEGF input configuration copied for each task.
    task_infos : List[dict]
        Relaxation task metadata including system atom boundaries.
    task_config : Dict[str, Any]
        Reserved per-task configuration for workflow compatibility.
    work_path : str, optional
        Directory initialized for subsequent DPNEGF output.

    Returns
    -------
    PrepNegfTasksResult
        Task names and modified DPNEGF input configurations.
    """
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
    """Run one DPNEGF task on a packed relaxed-system archive.

    Parameters
    ----------
    modified_negf_input_config : Dict[str, Any]
        Per-task config produced by ``dpnegf_prepare_negf_tasks``.
    task_name : str
        Name used for the output directory and returned task identifier.
    deeptb_model_path : Path
        DeePTB model used for transport calculations.
    relaxed_system_archive_path : Path
        Archive of relaxed VASP structures from the LAMMPS stage.
    negf_config : Dict[str, Any]
        Workflow options, including optional common self-energy reuse.
    work_path : str, optional
        Parent directory for calculation output.

    Returns
    -------
    RunDpnegfResult
        Log paths, transport-result paths, and archives of auxiliary plots/reports.
    """
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
    """Run ABACUS get_S for one relaxed POSCAR and collect sparse overlap output.

    Parameters
    ----------
    poscar_file_path : Path
        Relaxed VASP/POSCAR structure.
    input_file_path : Path
        ABACUS ``INPUT`` configured to output the overlap matrix.
    pp_file_paths : List[Path]
        Pseudopotential files used to generate ``STRU``.
    orb_file_paths : List[Path]
        Numerical-orbital files used to generate ``STRU``.
    run_config : Dict[str, Any]
        Execution settings; ``command`` defaults to ``abacus``.
    work_path : str, optional
        Parent directory for ABACUS output.

    Returns
    -------
    AbacusOverlapResult
        Paths required by the overlap-conversion stage.
    """
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
    """Convert ABACUS sparse overlap CSR output to dftio ``overlaps.h5``.

    Parameters
    ----------
    stru_path : Path
        ABACUS ``STRU.cif`` output.
    input_path : Path
        ABACUS ``INPUT`` output copy.
    running_log_path : Path
        ABACUS ``running_nscf.log`` file.
    overlap_csr_path : Path
        ABACUS sparse overlap CSR file.
    work_path : str, optional
        Parent directory for the dftio conversion.

    Returns
    -------
    ConvertOverlapResult
        Path to the generated ``overlaps.h5`` file.
    """
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
    """Plot transmission spectrum from a DPNEGF ``negf.out.pth`` file.

    Parameters
    ----------
    negf_out_path : Path
        DPNEGF result containing ``uni_grid`` and ``T_avg``.
    label : bool, optional
        Annotate each plotted point with its energy and transmission value.
    work_path : str, optional
        Directory where the image is saved.
    image_name : str, optional
        Output filename; a timestamped PNG name is generated when omitted.

    Returns
    -------
    TransmissionPlotResult
        Path to the generated transmission image.
    """
    return _transmission_plot(
        negf_out_path=negf_out_path,
        label=label,
        work_path=work_path,
        image_name=image_name,
    )
