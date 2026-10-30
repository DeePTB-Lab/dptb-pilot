from pathlib import Path
from typing import List, Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import RunLammpsResult
from dptb_pilot.tools.modules.lammps.execution import execute_lammps


@mcp.tool()
def submit_lammps_task(
        in_lammps_file_path: Path,
        lammps_data_file_path: Path,
        deepmd_model_file_path: Optional[Path] = None,
        lmp_command: str = "lmp",
        output_path: Path = Path("."),
        task_name: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
) -> dict:
    """Submit a LAMMPS task locally and retain its inputs, log, and results.

    Parameters
    ----------
    in_lammps_file_path : Path
        LAMMPS input script. It is staged as ``in.lammps``.
    lammps_data_file_path : Path
        LAMMPS data file. It is staged as ``lammps.data``.
    deepmd_model_file_path : Path, optional
        DeepMD model copied into the task directory next to the input script.
    lmp_command : str, optional
        Executable command, which may include MPI launcher arguments.
    output_path : Path, optional
        Parent directory where task inputs, logs, and generated structures are saved.
    task_name : str, optional
        Output task directory name; a unique name is generated when omitted.
    extra_args : List[str], optional
        Additional arguments appended to the LAMMPS command.

    Returns
    -------
    dict
        Task directory, log path, optional relaxed data/VASP paths, and command.
    """
    return execute_lammps(
        in_lammps_file_path=in_lammps_file_path,
        lammps_data_file_path=lammps_data_file_path,
        deepmd_model_file_path=deepmd_model_file_path,
        lmp_command=lmp_command,
        output_path=output_path,
        task_name=task_name,
        extra_args=extra_args,
    )


@mcp.tool()
def run_lammps(
        in_lammps_file_path: Path,
        lammps_data_file_path: Path,
        deepmd_model_file_path: Optional[Path] = None,
        lmp_command: str = "lmp",
        output_path: Path = Path("."),
        task_name: Optional[str] = None,
) -> RunLammpsResult:
    """Run one LAMMPS task and return the generated relaxed VASP structure.

    Parameters
    ----------
    in_lammps_file_path : Path
        LAMMPS input script, which must write ``relaxed.data``.
    lammps_data_file_path : Path
        LAMMPS data file referenced as ``lammps.data`` by the input script.
    deepmd_model_file_path : Path, optional
        DeepMD model copied into the run directory.
    lmp_command : str, optional
        LAMMPS executable command, optionally including MPI launcher arguments.
    output_path : Path, optional
        Parent directory for persistent run files and results.
    task_name : str, optional
        Output task directory name; a unique name is generated when omitted.

    Returns
    -------
    RunLammpsResult
        Path to the generated relaxed VASP structure.
    """
    result = execute_lammps(
        in_lammps_file_path=in_lammps_file_path,
        lammps_data_file_path=lammps_data_file_path,
        deepmd_model_file_path=deepmd_model_file_path,
        lmp_command=lmp_command,
        output_path=output_path,
        task_name=task_name,
    )
    structure_path = result["relaxed_structure_file_path"]
    if structure_path is None:
        raise FileNotFoundError(
            f"LAMMPS completed but did not create relaxed.data in {result['task_path']}"
        )
    return {"relaxed_system_file_path": structure_path}
