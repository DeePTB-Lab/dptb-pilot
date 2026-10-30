from pathlib import Path
from typing import Optional, TypedDict, List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import (
    _generate_lammps_input_file,
    _generate_uniaxial_strain_ensemble_lammps_input_file,
    _generate_uniaxial_strain_lammps_input_file,
    _generate_unstrained_lammps_input_file,
)

class GenerateUniaxialStrainInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]


class GenerateUniaxialStrainEnsembleInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]
    task_infos: List[dict]


class GenerateUnstrainedLammpsInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]
    task_paths: List[Path]


class GenerateLammpsInputResult(TypedDict):
    in_lammps_file_paths: List[Path]
    lammps_data_file_paths: List[Path]
    task_paths: List[Path]
    task_infos: List[dict]


@mcp.tool()
def generate_lammps_input_file(
        struture_file_path: Path,
        strains: Optional[List[float]] = None,
        lammps_run_method: str = 'relax',
        axis='auto',
        target_length=20,
        model_name='model.pb',
        temps: Optional[List[float]] = None,
        press: Optional[List[float]] = None,
        ensemble='smart',
        dt=1.0,
        nsteps=5000,
        temp_0=50,
        nstep_heating=10000,
        nstep_equil=10000,
        n_out=10,
        sample_interval=1000,
) -> GenerateLammpsInputResult:
    """
    Generate LAMMPS input/data files with one unified task selector.

    ``strains=None`` generates one unstrained structure. Passing a list of strain
    percentages generates one strained structure for each value.

    ``lammps_run_method`` selects the LAMMPS input workflow:
    - ``single``/``run0``: static run 0
    - ``relax``/``min``: energy minimization
    - ``ensemble``: use the ``ensemble`` argument
    - ``smart``/``nvt``/``npt``/``nve``: shorthand ensemble sampling modes

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure path. The misspelled parameter name is kept
        for API compatibility.
    strains : List[float], optional
        Strain percentages. ``None`` creates one unstrained task.
    lammps_run_method : str, optional
        ``single``, ``relax``, ``ensemble``, or an ensemble shorthand described above.
    axis : str or int, optional
        Supercell/strain axis: ``auto``, ``x``, ``y``, ``z``, or its integer index.
    target_length : float, optional
        Target supercell length along the selected axis in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by generated LAMMPS inputs.
    temps : List[float], optional
        Temperatures in K for ensemble tasks; defaults to ``[300]``.
    press : List[float], optional
        Pressures in bar for ensemble tasks; defaults to ``[0]``.
    ensemble : str, optional
        Ensemble used when ``lammps_run_method`` is ``ensemble``.
    dt : float, optional
        MD timestep in ps.
    nsteps : int, optional
        MD step count or minimization iteration limit.
    temp_0 : float, optional
        Initial temperature in K for smart heating.
    nstep_heating : int, optional
        Heating steps in the smart workflow.
    nstep_equil : int, optional
        Equilibration steps in the smart workflow.
    n_out : int, optional
        Number of sampled structures requested by the smart workflow.
    sample_interval : int, optional
        Timestep interval between sampled structures.

    Returns
    -------
    GenerateLammpsInputResult
        Generated input/data paths, task directories, and task metadata.
    """
    return _generate_lammps_input_file(
        output_root='.',
        poscar_file=struture_file_path,
        strains=strains,
        lammps_run_method=lammps_run_method,
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
def generate_uniaxial_strain_input_file(
        struture_file_path:Path,
        strain_list:List[float],
        axis='auto',
        target_length=20,
        model_name='model.pb',
        relax_after_strain=True
) -> GenerateUniaxialStrainInputResult:
    """
    Generate LAMMPS input files for uniaxial strain calculations along a specified axis.

    This tool creates a series of strained structures by applying uniaxial strain to a
    given crystal structure. For each strain value in the provided list, it:
    1. Builds a supercell along the specified axis to achieve a target length
    2. Applies the specified uniaxial strain
    3. Creates a separate directory with:
       - POSCAR file of the strained structure
       - LAMMPS data file (lammps.data) with atomic positions and cell information
       - LAMMPS input script (in.lammps) configured for relaxation with a machine learning potential

    The generated files are organized in subdirectories named by strain percentage
    (e.g., 'strain_+1.00pct/', 'strain_-0.50pct/') for easy management of multiple
    calculations.

    Parameters
    ----------
    struture_file_path : Path
        Path to the input structure file (POSCAR format). The file should contain
        the crystal structure to be strained.

    strain_list : List[float]
        List of strain values to apply, given as percentages. Positive values
        indicate tensile strain, negative values indicate compressive strain.
        Example: [1.0, 0.5, 0.0, -0.5, -1.0] for ±1% strain.

    axis : str or int, optional (default='auto')
        Crystallographic axis along which to apply strain. Options:
        - 'auto': Automatically select the axis that gives the desired target length
        - 'x', 'y', 'z': Manually specify the axis index ('x' for x-axis, 'y' for y-axis, 'z' for z-axis)

    target_length : float, optional (default=20)
        Target length in Angstroms for the supercell along the strain axis.
        The tool will create a supercell that approximates this length to ensure
        sufficient periodic length for accurate strain calculations.

    model_name : str, optional (default='model.pb')
        Name of the machine learning potential file (in PB format) to be used
        in the LAMMPS calculations. This file should be present in the working
        directory or specified with full path.

    relax_after_strain : bool, optional (default=True)
        Whether to include relaxation commands in the generated LAMMPS input script.
        If True, the LAMMPS script will perform structural relaxation after applying
        strain. If False, only a single-point calculation will be performed.

    Returns
    -------
    GenerateUniaxialStrainInputResult
        A dictionary containing:
        - in_lammps_file_paths: List of paths to generated LAMMPS input files (in.lammps)
          for each strain value
        - lammps_data_file_paths: List of paths to generated LAMMPS data files (lammps.data)
          for each strain value

    Notes
    -----
    The tool performs the following steps:
    1. Reads the input structure from the POSCAR file
    2. Builds an appropriate supercell to achieve the target length along the strain axis
    3. For each strain value in strain_list:
       a. Creates a subdirectory named 'strain_{value:+.2f}pct'
       b. Applies the uniaxial strain by scaling the lattice vectors
       c. Writes the strained structure to POSCAR format
       d. Converts to LAMMPS data format with proper element ordering
       e. Generates a LAMMPS input script with the specified potential

    The generated LAMMPS input scripts are configured to use the specified machine
    learning potential and include appropriate settings for accurate structure
    relaxation.

    Example
    -------
    >>> from pathlib import Path
    >>> result = generate_uniaxial_strain_input_file(
    ...     struture_file_path=Path("./POSCAR"),
    ...     strain_list=[-1.0, -0.5, 0.0, 0.5, 1.0],
    ...     axis='z',  # strain along z-axis
    ...     target_length=25.0,
    ...     model_name="my_potential.pb",
    ...     relax_after_strain=True
    ... )
    >>> print(f"Generated {len(result['in_lammps_file_paths'])} input files")
    """

    return _generate_uniaxial_strain_lammps_input_file(output_root='.',
                                                poscar_file=struture_file_path,
                                                strain_list=strain_list,
                                                axis=axis,
                                                target_length=target_length,
                                                model_name=model_name,
                                                relax_after_strain=relax_after_strain)


@mcp.tool()
def generate_uniaxial_strain_ensemble_input_file(
        struture_file_path: Path,
        strain_list: List[float],
        temps: List[float],
        press: List[float],
        ensemble='smart',
        axis='auto',
        target_length=20,
        model_name='model.pb',
        dt=1.0,
        nsteps=5000,
        temp_0=50,
        nstep_heating=10000,
        nstep_equil=10000,
        n_out=10,
        sample_interval=1000,
) -> GenerateUniaxialStrainEnsembleInputResult:
    """
    Generate LAMMPS input files for uniaxial-strain ensemble sampling tasks.

    For every strain, temperature, and pressure combination, this tool creates a
    task directory containing POSCAR, lammps.data, and in.lammps. The default
    ``smart`` ensemble writes a minimization, heating, equilibration, and
    trajectory-sampling workflow for ensemble-average structure sampling.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure path. The misspelled parameter name is kept
        for API compatibility.
    strain_list : List[float]
        Uniaxial strain percentages.
    temps : List[float]
        Temperatures in K; one task is generated per strain/temperature/pressure tuple.
    press : List[float]
        Pressures in bar.
    ensemble : str, optional
        LAMMPS ensemble: ``smart``, ``min``, ``nvt``, ``npt``, or ``nve``.
    axis : str or int, optional
        Supercell/strain axis: ``auto``, ``x``, ``y``, ``z``, or its integer index.
    target_length : float, optional
        Target supercell length along the selected axis in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by generated inputs.
    dt : float, optional
        MD timestep in ps.
    nsteps : int, optional
        MD step count for direct ensemble workflows.
    temp_0 : float, optional
        Initial temperature in K for smart heating.
    nstep_heating : int, optional
        Heating steps in the smart workflow.
    nstep_equil : int, optional
        Equilibration steps in the smart workflow.
    n_out : int, optional
        Number of sampled structures requested by the smart workflow.
    sample_interval : int, optional
        Timestep interval between sampled structures.

    Returns
    -------
    GenerateUniaxialStrainEnsembleInputResult
        Generated input/data paths and metadata for every task.
    """
    return _generate_uniaxial_strain_ensemble_lammps_input_file(
        output_root='.',
        poscar_file=struture_file_path,
        strain_list=strain_list,
        temps=temps,
        press=press,
        ensemble=ensemble,
        axis=axis,
        target_length=target_length,
        model_name=model_name,
        dt=dt,
        nsteps=nsteps,
        temp_0=temp_0,
        nstep_heating=nstep_heating,
        nstep_equil=nstep_equil,
        n_out=n_out,
        sample_interval=sample_interval,
    )


@mcp.tool()
def generate_unstrained_lammps_input_file(
        struture_file_path: Path,
        axis='auto',
        target_length=20,
        model_name='model.pb',
        relax=True,
) -> GenerateUnstrainedLammpsInputResult:
    """
    Generate an unstrained LAMMPS task directory with POSCAR, lammps.data, and in.lammps.

    The structure is expanded with the same supercell convention used by the
    uniaxial strain tools, but no strain is applied.

    Parameters
    ----------
    struture_file_path : Path
        Input POSCAR/VASP structure path. The misspelled parameter name is kept
        for API compatibility.
    axis : str or int, optional
        Supercell axis: ``auto``, ``x``, ``y``, ``z``, or its integer index.
    target_length : float, optional
        Target supercell length along the selected axis in Angstrom.
    model_name : str, optional
        DeepMD model filename referenced by the generated LAMMPS input.
    relax : bool, optional
        Generate minimization commands when true, otherwise a static run.

    Returns
    -------
    GenerateUnstrainedLammpsInputResult
        Generated input/data paths and the unstrained task directory.
    """
    return _generate_unstrained_lammps_input_file(
        output_root='.',
        poscar_file=struture_file_path,
        axis=axis,
        target_length=target_length,
        model_name=model_name,
        relax=relax,
    )


