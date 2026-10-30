from pathlib import Path
from typing import TypedDict

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import RunLammpsResult
from dptb_pilot.tools.modules.deeptb.submodules.lammps import _run_lammps

@mcp.tool()
def run_lammps(
        in_lammps_file_path: Path,
        lammps_data_file_path: Path,
        deepmd_model_file_path: Path = None,
        lmp_command: str = 'lmp'
) -> RunLammpsResult:
    """
    Run a LAMMPS relaxation task and save the relaxed structure as VASP format.

    Parameters
    ----------
    in_lammps_file_path : Path
        Path to the LAMMPS input script. The run expects this script to write
        ``relaxed.data``.
    lammps_data_file_path : Path
        Path to the LAMMPS data file used by the input script.
    deepmd_model_file_path : Path, optional
        DeepMD model file to copy into the temporary run directory.
    lmp_command : str, optional
        LAMMPS executable command, by default ``lmp``.

    Returns
    -------
    RunLammpsResult
        Dictionary with ``relaxed_system_file_path`` pointing to the generated
        relaxed VASP structure.
    """

    return _run_lammps(in_lammps_file_path=in_lammps_file_path,
                       lammps_data_file_path=lammps_data_file_path,
                       deepmd_model_file_path=deepmd_model_file_path,
                       lmp_command=lmp_command)