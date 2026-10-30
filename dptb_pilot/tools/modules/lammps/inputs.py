from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import _generate_lammps_input_file


def _generate_for_method(
        structure_file_path: Path,
        output_path: Path,
        method: str,
        strains: Optional[List[float]],
        axis: Union[str, int],
        target_length: float,
        model_name: str,
        temps: Optional[List[float]],
        press: Optional[List[float]],
        ensemble: str,
        dt: float,
        nsteps: int,
        temp_0: float,
        nstep_heating: int,
        nstep_equil: int,
        n_out: int,
        sample_interval: int,
) -> Dict[str, Any]:
    return _generate_lammps_input_file(
        output_root=output_path,
        poscar_file=structure_file_path,
        strains=strains,
        lammps_run_method=method,
        axis=axis,
        target_length=target_length,
        model_name=model_name,
        temps=temps,
        press=press,
        ensemble=ensemble,
        dt=dt,
        nsteps=nsteps,
        temp_0=temp_0,
        nstep_heating=nstep_heating,
        nstep_equil=nstep_equil,
        n_out=n_out,
        sample_interval=sample_interval,
    )


@mcp.tool()
def generate_lammps_inputs(
        structure_file_path: Path,
        output_path: Path = Path("."),
        methods: Optional[List[str]] = None,
        strains: Optional[List[float]] = None,
        axis: Union[str, int] = "auto",
        target_length: float = 20.0,
        model_name: str = "model.pb",
        temps: Optional[List[float]] = None,
        press: Optional[List[float]] = None,
        ensemble: str = "smart",
        dt: float = 0.001,
        nsteps: int = 5000,
        temp_0: float = 50.0,
        nstep_heating: int = 10000,
        nstep_equil: int = 10000,
        n_out: int = 10,
        sample_interval: int = 1000,
) -> Dict[str, Any]:
    """Generate LAMMPS input and data files for one or more run methods.

    Parameters
    ----------
    structure_file_path : Path
        Input POSCAR/VASP structure.
    output_path : Path, optional
        Directory that receives the generated task directories.
    methods : List[str], optional
        Methods to generate: ``single``, ``relax``, ``min``, ``ensemble``,
        ``smart``, ``nvt``, ``npt``, or ``nve``. Defaults to ``["relax"]``.
    strains : List[float], optional
        Uniaxial strain percentages. Omit to generate one unstrained structure.
    axis : str or int, optional
        Supercell and strain axis: ``auto``, ``x``, ``y``, ``z``, or 0-2.
    target_length : float, optional
        Target supercell length along the selected axis, in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by the generated input scripts.
    temps : List[float], optional
        Temperatures in K for ensemble tasks; defaults to 300 K.
    press : List[float], optional
        Pressures in bar for ensemble tasks; defaults to 0 bar.
    ensemble : str, optional
        Ensemble selected by the ``ensemble`` method.
    dt : float, optional
        MD timestep in ps.
    nsteps : int, optional
        MD run steps or minimization iteration limit.
    temp_0 : float, optional
        Initial temperature in K for smart heating.
    nstep_heating : int, optional
        Heating steps for the smart workflow.
    nstep_equil : int, optional
        Equilibration steps for the smart workflow.
    n_out : int, optional
        Number of smart-workflow configurations to sample.
    sample_interval : int, optional
        Steps between smart-workflow trajectory frames.

    Returns
    -------
    dict
        Generated task paths, input/data paths, and metadata grouped by method.
    """
    structure_file_path = Path(structure_file_path)
    if not structure_file_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_methods = methods or ["relax"]
    if not selected_methods:
        raise ValueError("methods must contain at least one LAMMPS method")
    normalized_methods = [str(method).lower() for method in selected_methods]
    if len(set(normalized_methods)) != len(normalized_methods):
        raise ValueError("methods cannot contain duplicate run methods")

    method_results: Dict[str, Any] = {}
    multi_method = len(normalized_methods) > 1
    model_filename = Path(model_name).name
    for normalized in normalized_methods:
        method_output = output_path / normalized if multi_method else output_path
        method_results[normalized] = _generate_for_method(
            structure_file_path,
            method_output,
            normalized,
            strains,
            axis,
            target_length,
            model_filename,
            temps,
            press,
            ensemble,
            dt,
            nsteps,
            temp_0,
            nstep_heating,
            nstep_equil,
            n_out,
            sample_interval,
        )
    return {"method_results": method_results}


@mcp.tool()
def generate_lammps_input_file(
        struture_file_path: Path,
        output_path: Path = Path("."),
        strains: Optional[List[float]] = None,
        lammps_run_method: str = "relax",
        axis: Union[str, int] = "auto",
        target_length: float = 20.0,
        model_name: str = "model.pb",
        temps: Optional[List[float]] = None,
        press: Optional[List[float]] = None,
        ensemble: str = "smart",
        dt: float = 0.001,
        nsteps: int = 5000,
        temp_0: float = 50.0,
        nstep_heating: int = 10000,
        nstep_equil: int = 10000,
        n_out: int = 10,
        sample_interval: int = 1000,
) -> Dict[str, Any]:
    """Generate input/data files for a single LAMMPS run method.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure.
    output_path : Path, optional
        Directory that receives the generated task directories.
    strains : List[float], optional
        Uniaxial strain percentages; omit for an unstrained structure.
    lammps_run_method : str, optional
        ``single``, ``relax``, ``ensemble``, ``smart``, ``nvt``, ``npt``, or ``nve``.
    axis : str or int, optional
        Supercell and strain axis: ``auto``, ``x``, ``y``, ``z``, or 0-2.
    target_length : float, optional
        Target supercell length along the selected axis, in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by the generated input script.
    temps : List[float], optional
        Temperatures in K for ensemble runs.
    press : List[float], optional
        Pressures in bar for ensemble runs.
    ensemble : str, optional
        Ensemble selected by ``lammps_run_method="ensemble"``.
    dt : float, optional
        MD timestep in ps.
    nsteps : int, optional
        MD run steps or minimization iteration limit.
    temp_0 : float, optional
        Initial temperature in K for smart heating.
    nstep_heating : int, optional
        Heating steps for the smart workflow.
    nstep_equil : int, optional
        Equilibration steps for the smart workflow.
    n_out : int, optional
        Number of smart-workflow configurations to sample.
    sample_interval : int, optional
        Steps between smart-workflow trajectory frames.

    Returns
    -------
    dict
        Generated task paths, input/data paths, and task metadata.
    """
    result = generate_lammps_inputs(
        structure_file_path=struture_file_path,
        output_path=output_path,
        methods=[lammps_run_method],
        strains=strains,
        axis=axis,
        target_length=target_length,
        model_name=model_name,
        temps=temps,
        press=press,
        ensemble=ensemble,
        dt=dt,
        nsteps=nsteps,
        temp_0=temp_0,
        nstep_heating=nstep_heating,
        nstep_equil=nstep_equil,
        n_out=n_out,
        sample_interval=sample_interval,
    )
    return result["method_results"][lammps_run_method.lower()]


@mcp.tool()
def generate_uniaxial_strain_input_file(
        struture_file_path: Path,
        strain_list: List[float],
        axis: Union[str, int] = "auto",
        target_length: float = 20.0,
        model_name: str = "model.pb",
        relax_after_strain: bool = True,
) -> Dict[str, Any]:
    """Generate one LAMMPS task for each requested uniaxial strain.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure; the historical parameter spelling is retained.
    strain_list : List[float]
        Strain percentages to apply along ``axis``.
    axis : str or int, optional
        Supercell and strain axis: ``auto``, ``x``, ``y``, ``z``, or 0-2.
    target_length : float, optional
        Target supercell length along the selected axis, in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by each input script.
    relax_after_strain : bool, optional
        Use energy minimization when true, or a static run when false.

    Returns
    -------
    dict
        Input/data file paths and task metadata for each strain.
    """
    return _generate_for_method(
        Path(struture_file_path), Path("."), "relax" if relax_after_strain else "single",
        strain_list, axis, target_length, Path(model_name).name, None, None,
        "smart", 0.001, 5000, 50.0, 10000, 10000, 10, 1000,
    )


@mcp.tool()
def generate_uniaxial_strain_ensemble_input_file(
        struture_file_path: Path,
        strain_list: List[float],
        temps: List[float],
        press: List[float],
        ensemble: str = "smart",
        axis: Union[str, int] = "auto",
        target_length: float = 20.0,
        model_name: str = "model.pb",
        dt: float = 0.001,
        nsteps: int = 5000,
        temp_0: float = 50.0,
        nstep_heating: int = 10000,
        nstep_equil: int = 10000,
        n_out: int = 10,
        sample_interval: int = 1000,
) -> Dict[str, Any]:
    """Generate strain, temperature, and pressure combinations for LAMMPS.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure; the historical parameter spelling is retained.
    strain_list : List[float]
        Strain percentages to apply along ``axis``.
    temps : List[float]
        Temperatures in K.
    press : List[float]
        Pressures in bar.
    ensemble : str, optional
        Run style: ``smart``, ``min``, ``nvt``, ``npt``, or ``nve``.
    axis : str or int, optional
        Supercell and strain axis: ``auto``, ``x``, ``y``, ``z``, or 0-2.
    target_length : float, optional
        Target supercell length along the selected axis, in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by each input script.
    dt : float, optional
        MD timestep in ps.
    nsteps : int, optional
        Run steps or minimization iteration limit.
    temp_0 : float, optional
        Initial temperature in K for smart heating.
    nstep_heating : int, optional
        Heating steps for the smart workflow.
    nstep_equil : int, optional
        Equilibration steps for the smart workflow.
    n_out : int, optional
        Number of smart-workflow configurations to sample.
    sample_interval : int, optional
        Steps between smart-workflow trajectory frames.

    Returns
    -------
    dict
        Input/data paths and task metadata for every strain/temperature/pressure tuple.
    """
    return _generate_for_method(
        Path(struture_file_path), Path("."), "ensemble", strain_list, axis,
        target_length, Path(model_name).name, temps, press, ensemble, dt, nsteps,
        temp_0, nstep_heating, nstep_equil, n_out, sample_interval,
    )


@mcp.tool()
def generate_unstrained_lammps_input_file(
        struture_file_path: Path,
        axis: Union[str, int] = "auto",
        target_length: float = 20.0,
        model_name: str = "model.pb",
        relax: bool = True,
) -> Dict[str, Any]:
    """Generate one LAMMPS task for an unstrained structure.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure; the historical parameter spelling is retained.
    axis : str or int, optional
        Supercell axis: ``auto``, ``x``, ``y``, ``z``, or 0-2.
    target_length : float, optional
        Target supercell length along the selected axis, in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by the generated input.
    relax : bool, optional
        Use energy minimization when true, or a static run when false.

    Returns
    -------
    dict
        Input/data paths and task directory for the unstrained structure.
    """
    return _generate_for_method(
        Path(struture_file_path), Path("."), "relax" if relax else "single", None,
        axis, target_length, Path(model_name).name, None, None, "smart", 0.001,
        5000, 50.0, 10000, 10000, 10, 1000,
    )
