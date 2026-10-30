from pathlib import Path
from typing import Any, Dict, List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.dpnegf.submodules.lammps import prepare_lammps_tasks as _prepare_lammps_tasks


@mcp.tool()
def prepare_lammps_tasks(
        stacked_system_paths: List[Path],
        system_infos: List[dict],
        relax_config: Dict[str, Any],
        inputs_config: Dict[str, Any],
        work_path: str = ".",
) -> Dict[str, Any]:
    """Prepare LAMMPS relaxation tasks for structures with optional fixed regions.

    Parameters
    ----------
    stacked_system_paths : List[Path]
        Input structures, such as POSCAR files for stacked/device systems.
    system_infos : List[dict]
        Per-structure atom counts and ``atom_index`` region boundaries.
    relax_config : Dict[str, Any]
        Ensemble, temperatures, pressures, timestep, run length, and optional
        fixed-device radius and smart-workflow settings.
    inputs_config : Dict[str, Any]
        ``deepmd_model_path`` and optional ``deepmd_model_type_map`` element order.
    work_path : str, optional
        Parent directory for generated task folders.

    Returns
    -------
    Dict[str, Any]
        Task root, directories, names, and metadata.
    """
    if len(stacked_system_paths) != len(system_infos):
        raise ValueError("stacked_system_paths and system_infos must have equal lengths")
    return _prepare_lammps_tasks(
        stacked_system_paths=stacked_system_paths,
        system_infos=system_infos,
        relax_config=relax_config,
        inputs_config=inputs_config,
        work_path=work_path,
    )
