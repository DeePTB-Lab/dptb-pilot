from pathlib import Path
from typing import List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import PressTubeTaskResult
from dptb_pilot.tools.modules.deeptb.submodules.press_tube import build_and_generate


@mcp.tool()
def generate_press_tube_lammps_tasks(
        poscar_path: Path,
        out_root: str = "tasks_indents",
        target_length: float = 100.0,
        axis: str = "auto",
        n_repeat: int = None,
        indenter_radius: float = 4.0,
        indenter_height_factor: float = 1.2,
        indent_depths: List[float] = None,
        indenter_spacing: float = 1.42,
        indenter_symbol: str = "C",
        deepmd_model_name: str = "MODEL.pb",
        ensemble: str = "min",
        dt: float = 1.0,
        nsteps: int = 5000
) -> PressTubeTaskResult:
    """
    Generate LAMMPS indentation tasks for pressing a tube-like structure.

    The tool builds a supercell from a POSCAR/VASP structure, creates cylindrical
    indenter atoms for each lateral depth, and writes one ``lammps.data`` plus
    ``in.lammps`` task directory per indentation case.

    Parameters
    ----------
    poscar_path : Path
        Input POSCAR/VASP structure path readable by ASE.
    out_root : str, optional
        Root directory where indentation task folders are created.
    target_length : float, optional
        Target tube-axis supercell length in Angstrom.
    axis : str, optional
        Tube axis selection: ``auto``, ``x``, ``y``, or ``z``.
    n_repeat : int, optional
        Explicit repeat count along the selected axis. Overrides ``target_length``.
    indenter_radius : float, optional
        Radius of the cylindrical indenter in Angstrom.
    indenter_height_factor : float, optional
        Indenter height as a multiple of the tube-axis cell length.
    indent_depths : List[float], optional
        Lateral offsets for generated indentation tasks. Smaller values press
        deeper toward the tube center.
    indenter_spacing : float, optional
        Approximate spacing between generated indenter atoms.
    indenter_symbol : str, optional
        Chemical symbol used for indenter atoms.
    deepmd_model_name : str, optional
        DeepMD model filename referenced by generated LAMMPS scripts.
    ensemble : str, optional
        Relaxation/dynamics mode: ``min``, ``nvt``, or ``nve``.
    dt : float, optional
        MD timestep for non-minimization ensembles.
    nsteps : int, optional
        Minimization iteration cap or MD step count.

    Returns
    -------
    PressTubeTaskResult
        Root directory, number of generated task directories, and their paths.
    """
    build_and_generate(
        poscar_path=str(poscar_path),
        out_root=out_root,
        target_length=target_length,
        axis=axis,
        n_repeat=n_repeat,
        indenter_radius=indenter_radius,
        indenter_height_factor=indenter_height_factor,
        indent_depths=indent_depths,
        indenter_spacing=indenter_spacing,
        indenter_symbol=indenter_symbol,
        deepmd_model_name=deepmd_model_name,
        ensemble=ensemble,
        dt=dt,
        nsteps=nsteps,
    )
    task_root = Path(out_root).absolute()
    task_paths = sorted(p for p in task_root.iterdir() if p.is_dir())
    return {
        "task_root_path": task_root,
        "task_count": len(task_paths),
        "task_paths": task_paths,
    }
