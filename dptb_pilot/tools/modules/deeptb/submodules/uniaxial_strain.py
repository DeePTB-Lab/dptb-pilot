"""
Generate CNT supercell and apply uniaxial strain
consistent with Yang & Han PRL 1999 definition:

ε = (L - L0) / L0
"""
import os
from pathlib import Path
from typing import List

import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ase.io import read, write

from dptb_pilot.tools.modules.deeptb.submodules.supercell import build_supercell


def apply_uniaxial_strain(atoms, axis, strain_percent):
    """
    Apply affine uniaxial strain along axis
    """
    eps = strain_percent / 100.0

    cell = atoms.get_cell()
    new_cell = cell.copy()

    new_cell[axis] *= (1.0 + eps)

    atoms.set_cell(new_cell, scale_atoms=True)

    return atoms


def write_lammps_input(task_dir, symbols, relax=False, model_name='model.pb'):
    """
    symbols: unique chemical symbols in specorder order
    """
    if relax:
        run_block = _ensemble_block("min", nsteps=10000)
    else:
        run_block = "thermo          100\nrun             0"

    _write_lammps_input(
        task_dir=task_dir,
        symbols=symbols,
        model_name=model_name,
        run_block=run_block,
    )


def _build_specorder(atoms):
    symbols = []
    for symbol in atoms.get_chemical_symbols():
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _mass_lines(symbols):
    lines = []
    for i, sym in enumerate(symbols, start=1):
        z = atomic_numbers[sym]
        mass = atomic_masses[z]
        lines.append(f"mass {i} {mass:.6f}  # {sym}")
    return "\n".join(lines)


def _write_lammps_input(task_dir, symbols, model_name, run_block):
    task_dir = Path(task_dir)

    with open(task_dir / "in.lammps", "w", encoding="utf-8") as f:

        f.write(f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       lammps.data

{_mass_lines(symbols)}

pair_style      deepmd {model_name}
pair_coeff      * * {' '.join(symbols)}

neighbor        2.0 bin
neigh_modify    every 10 delay 0 check yes

{run_block}

write_data      relaxed.data
""")


def _ensemble_block(ensemble,
                    T=300,
                    P=0,
                    dt=1.0,
                    nsteps=5000,
                    temp_0=50,
                    nstep_heating=10000,
                    nstep_equil=10000,
                    n_out=10,
                    sample_interval=1000,
                    group_name="all"):
    ensemble = ensemble.lower()
    if ensemble == "min":
        return (
            "thermo          100\n"
            "min_style       cg\n"
            f"minimize        1e-6 1e-8 1000 {nsteps}"
        )
    if ensemble == "nvt":
        return (
            f"velocity        {group_name} create {T} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} nvt temp {T} {T} 0.1\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}\n"
            "unfix           1"
        )
    if ensemble == "npt":
        return (
            f"velocity        {group_name} create {T} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} npt temp {T} {T} 0.1 iso {P:.6f} {P:.6f} 1.0\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}\n"
            "unfix           1"
        )
    if ensemble == "nve":
        return (
            f"velocity        {group_name} create {T} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} nve\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}\n"
            "unfix           1"
        )
    if ensemble == "smart":
        sampling_steps = max(int(sample_interval) * max(int(n_out) - 1, 0), 0)
        return "\n".join([
            "# 0. energy minimization",
            "min_style       cg",
            "minimize        1e-10 1e-10 10000 10000",
            "reset_timestep  0",
            "",
            "# 1. low-temperature velocity initialization",
            f"velocity        {group_name} create {temp_0} 12345 mom yes rot yes dist gaussian",
            "",
            "# 2. heating",
            f"fix             heat {group_name} nvt temp {temp_0} {T} 0.1",
            "thermo          100",
            f"timestep        {dt}",
            f"run             {nstep_heating}",
            "unfix           heat",
            "reset_timestep  0",
            "",
            "# 3. equilibration",
            f"fix             equil {group_name} nvt temp {T} {T} 0.1",
            f"dump            eq all atom {sample_interval} equil.xyz",
            f"run             {nstep_equil}",
            "unfix           equil",
            "undump          eq",
            "reset_timestep  0",
            "",
            "# 4. ensemble-average sampling",
            f"fix             sample {group_name} nvt temp {T} {T} 0.1",
            f"dump            traj all custom {sample_interval} traj.xyz id type xu yu zu",
            'dump_modify     traj format line "%d %d %.16f %.16f %.16f"',
            f"run             {sampling_steps}",
            "unfix           sample",
            "undump          traj",
        ])
    raise ValueError("ensemble must be one of 'min', 'nvt', 'npt', 'nve', or 'smart'")

def _generate_uniaxial_strain_structure(output_root,
                                        poscar_file:Path,
                                        strain_list:List[float],
                                        axis='auto',
                                        target_length=20):
    """
    Generate POSCAR/VASP structure files for a list of uniaxial strain values.

    Parameters
    ----------
    output_root : str or Path
        Directory where strained structure files are written.
    poscar_file : Path
        Input POSCAR/VASP structure file.
    strain_list : List[float]
        Strain percentages to apply. Positive values are tensile and negative
        values are compressive.
    axis : str, optional
        Axis to strain: ``auto``, ``x``, ``y``, or ``z``.
    target_length : float, optional
        Target supercell length along the selected axis before applying strain.

    Returns
    -------
    dict
        Dictionary with ``poscar_file_paths`` containing generated structure paths.
    """
    OUTPUT_ROOT_PATH = Path(output_root)
    os.makedirs(OUTPUT_ROOT_PATH, exist_ok=True)

    strain_list = [float(strain) for strain in strain_list]

    atoms = read(poscar_file)

    supercell, axis, _ = build_supercell(atoms,
                                         axis=axis,
                                         target_length=target_length)

    L0 = np.linalg.norm(supercell.get_cell()[axis])
    print(f"Reference length L0 = {L0:.3f} Å")

    poscar_file_paths = []

    for strain in strain_list:
        strained_atoms = supercell.copy()
        strained_atoms = apply_uniaxial_strain(
            strained_atoms, axis, strain
        )

        # Write POSCAR
        file_name = str(poscar_file.name).split('.')[0] + f"{strain:+.2f}.vasp"
        write(OUTPUT_ROOT_PATH / file_name, strained_atoms, format="vasp")
        poscar_file_paths.append(Path(OUTPUT_ROOT_PATH / file_name))

        print(f"Generated strain {strain:+.2f}%")

    print("All strain structures generated.")
    return {"poscar_file_paths":poscar_file_paths}

def _generate_uniaxial_strain_lammps_input_file(output_root,
                                                poscar_file:Path,
                                                strain_list:List[float],
                                                axis='auto',
                                                target_length=20,
                                                model_name='model.pb',
                                                relax_after_strain=True):
    """
    Generate strained structures and matching LAMMPS input/data files.

    Parameters
    ----------
    output_root : str or Path
        Root directory where one subdirectory per strain value is created.
    poscar_file : Path
        Input POSCAR/VASP structure file.
    strain_list : List[float]
        Strain percentages to apply.
    axis : str, optional
        Axis to strain: ``auto``, ``x``, ``y``, or ``z``.
    target_length : float, optional
        Target supercell length along the selected axis.
    model_name : str, optional
        DeepMD model filename referenced in generated LAMMPS input scripts.
    relax_after_strain : bool, optional
        Whether generated scripts should relax the strained structure.

    Returns
    -------
    dict
        ``in_lammps_file_paths`` and ``lammps_data_file_paths`` lists for all
        generated strain tasks.
    """
    OUTPUT_ROOT_PATH = Path(output_root)
    OUTPUT_ROOT_PATH.mkdir(exist_ok=True)

    strain_list = [float(strain) for strain in strain_list]

    atoms = read(poscar_file)

    supercell, axis, _ = build_supercell(atoms,
                                         axis=axis,
                                         target_length=target_length)

    L0 = np.linalg.norm(supercell.get_cell()[axis])
    print(f"Reference length L0 = {L0:.3f} Å")

    # 获取元素顺序（用于 specorder）
    symbols = _build_specorder(supercell)

    in_lammps_file_paths = []
    lammps_data_file_paths = []

    for strain in strain_list:
        task_dir = OUTPUT_ROOT_PATH / f"strain_{strain:+.2f}pct"
        task_dir.mkdir(exist_ok=True)

        strained_atoms = supercell.copy()
        strained_atoms = apply_uniaxial_strain(
            strained_atoms, axis, strain
        )

        # Write POSCAR
        write(task_dir / "POSCAR", strained_atoms)

        # Write LAMMPS data
        write(task_dir / "lammps.data",
              strained_atoms,
              format="lammps-data",
              specorder=symbols)
        lammps_data_file_paths.append(Path(task_dir / "lammps.data"))

        # Write LAMMPS input
        write_lammps_input(task_dir,
                           symbols=symbols,
                           relax=relax_after_strain,
                           model_name=model_name)
        in_lammps_file_paths.append(Path(task_dir / "in.lammps"))

        print(f"Generated strain {strain:+.2f}%")

    print("All strain structures generated.")
    return {"in_lammps_file_paths":in_lammps_file_paths, "lammps_data_file_paths":lammps_data_file_paths}


def _generate_uniaxial_strain_input_file(*args, **kwargs):
    """Backward-compatible name for the strain plus LAMMPS input generator."""
    return _generate_uniaxial_strain_lammps_input_file(*args, **kwargs)


def _generate_uniaxial_strain_ensemble_lammps_input_file(output_root,
                                                         poscar_file: Path,
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
                                                         sample_interval=1000):
    """
    Generate strained LAMMPS tasks for ensemble-average sampling.

    The generated ``smart`` ensemble input performs minimization, heating,
    equilibration, and trajectory sampling. Other ensembles write a direct
    minimization or MD run block with the requested temperature/pressure.
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    strain_list = [float(strain) for strain in strain_list]
    temps = [float(temp) for temp in temps]
    press = [float(pres) for pres in press]

    atoms = read(poscar_file)
    supercell, axis, _ = build_supercell(
        atoms,
        axis=axis,
        target_length=target_length,
    )

    L0 = np.linalg.norm(supercell.get_cell()[axis])
    print(f"Reference length L0 = {L0:.3f} A")

    symbols = _build_specorder(supercell)

    in_lammps_file_paths = []
    lammps_data_file_paths = []
    task_infos = []

    for strain in strain_list:
        strained_atoms = supercell.copy()
        strained_atoms = apply_uniaxial_strain(strained_atoms, axis, strain)

        for temp in temps:
            for pres in press:
                task_name = f"strain_{strain:+.2f}pct_{temp:g}K_{pres:g}bar"
                task_dir = output_root_path / task_name
                task_dir.mkdir(parents=True, exist_ok=True)

                write(task_dir / "POSCAR", strained_atoms)
                write(
                    task_dir / "lammps.data",
                    strained_atoms,
                    format="lammps-data",
                    specorder=symbols,
                )

                run_block = _ensemble_block(
                    ensemble=ensemble,
                    T=temp,
                    P=pres,
                    dt=dt,
                    nsteps=nsteps,
                    temp_0=temp_0,
                    nstep_heating=nstep_heating,
                    nstep_equil=nstep_equil,
                    n_out=n_out,
                    sample_interval=sample_interval,
                    group_name="all",
                )
                _write_lammps_input(
                    task_dir=task_dir,
                    symbols=symbols,
                    model_name=model_name,
                    run_block=run_block,
                )

                in_lammps_file_paths.append(Path(task_dir / "in.lammps"))
                lammps_data_file_paths.append(Path(task_dir / "lammps.data"))
                task_infos.append({
                    "strain": strain,
                    "temp": temp,
                    "pres": pres,
                    "ensemble": ensemble,
                    "axis": axis,
                    "reference_length": L0,
                })
                print(f"Generated ensemble sampling task: {task_name}")

    print("All ensemble sampling tasks generated.")
    return {
        "in_lammps_file_paths": in_lammps_file_paths,
        "lammps_data_file_paths": lammps_data_file_paths,
        "task_infos": task_infos,
    }


def _generate_unstrained_lammps_input_file(output_root,
                                           poscar_file: Path,
                                           axis='auto',
                                           target_length=20,
                                           model_name='model.pb',
                                           relax=True):
    """
    Generate an unstrained LAMMPS input/data task from a POSCAR/VASP structure.

    A supercell is built with the same axis/target-length convention used by the
    strain workflow, but no affine strain is applied.
    """
    output_root_path = Path(output_root)
    task_dir = output_root_path / "unstrained"
    task_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(poscar_file)
    supercell, axis, _ = build_supercell(
        atoms,
        axis=axis,
        target_length=target_length,
    )

    symbols = _build_specorder(supercell)

    write(task_dir / "POSCAR", supercell)
    write(
        task_dir / "lammps.data",
        supercell,
        format="lammps-data",
        specorder=symbols,
    )
    write_lammps_input(
        task_dir=task_dir,
        symbols=symbols,
        relax=relax,
        model_name=model_name,
    )

    print(f"Generated unstrained LAMMPS task -> {task_dir}")
    return {
        "in_lammps_file_paths": [Path(task_dir / "in.lammps")],
        "lammps_data_file_paths": [Path(task_dir / "lammps.data")],
        "task_paths": [Path(task_dir)],
    }


def _generate_lammps_input_file(output_root,
                                poscar_file: Path,
                                strains=None,
                                lammps_run_method='relax',
                                axis='auto',
                                target_length=20,
                                model_name='model.pb',
                                temps=None,
                                press=None,
                                ensemble='smart',
                                dt=0.001,
                                nsteps=5000,
                                temp_0=50,
                                nstep_heating=10000,
                                nstep_equil=10000,
                                n_out=10,
                                sample_interval=1000):
    """
    Generate LAMMPS tasks with optional uniaxial strain and selectable run method.

    ``strains=None`` creates one unstrained task. Otherwise, one structure is
    generated for every strain percentage in ``strains``.

    ``lammps_run_method`` accepts:
    - ``single``/``run0``: write a run-0 input
    - ``relax``/``min``: write a minimization input
    - ``ensemble``: use ``ensemble`` to select smart/nvt/npt/nve/min
    - ``smart``/``nvt``/``npt``/``nve``: shorthand ensemble methods
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    method = str(lammps_run_method).lower()
    ensemble_methods = {"ensemble", "smart", "nvt", "npt", "nve"}
    single_methods = {"single", "run0", "static"}
    relax_methods = {"relax", "min", "minimize"}

    if method in ensemble_methods:
        use_ensemble = True
        selected_ensemble = ensemble if method == "ensemble" else method
    elif method in single_methods:
        use_ensemble = False
        relax = False
        selected_ensemble = None
    elif method in relax_methods:
        use_ensemble = False
        relax = True
        selected_ensemble = None
    else:
        raise ValueError(
            "lammps_run_method must be one of single/run0/static, "
            "relax/min/minimize, ensemble, smart, nvt, npt, or nve"
        )

    temps = [300.0] if temps is None else [float(temp) for temp in temps]
    press = [0.0] if press is None else [float(pres) for pres in press]

    atoms = read(poscar_file)
    supercell, axis, _ = build_supercell(
        atoms,
        axis=axis,
        target_length=target_length,
    )
    L0 = np.linalg.norm(supercell.get_cell()[axis])
    print(f"Reference length L0 = {L0:.3f} A")

    symbols = _build_specorder(supercell)

    if strains is None:
        structure_items = [(None, "unstrained", supercell.copy())]
    else:
        structure_items = []
        for strain in [float(strain) for strain in strains]:
            strained_atoms = supercell.copy()
            strained_atoms = apply_uniaxial_strain(strained_atoms, axis, strain)
            structure_items.append((strain, f"strain_{strain:+.2f}pct", strained_atoms))

    in_lammps_file_paths = []
    lammps_data_file_paths = []
    task_paths = []
    task_infos = []

    for strain, structure_name, task_atoms in structure_items:
        if use_ensemble:
            temp_pres_pairs = ((temp, pres) for temp in temps for pres in press)
        else:
            temp_pres_pairs = ((None, None),)

        for temp, pres in temp_pres_pairs:
            if use_ensemble:
                task_name = f"{structure_name}_{temp:g}K_{pres:g}bar"
            else:
                task_name = structure_name

            task_dir = output_root_path / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            write(task_dir / "POSCAR", task_atoms)
            write(
                task_dir / "lammps.data",
                task_atoms,
                format="lammps-data",
                specorder=symbols,
            )

            if use_ensemble:
                run_block = _ensemble_block(
                    ensemble=selected_ensemble,
                    T=temp,
                    P=pres,
                    dt=dt,
                    nsteps=nsteps,
                    temp_0=temp_0,
                    nstep_heating=nstep_heating,
                    nstep_equil=nstep_equil,
                    n_out=n_out,
                    sample_interval=sample_interval,
                    group_name="all",
                )
                _write_lammps_input(
                    task_dir=task_dir,
                    symbols=symbols,
                    model_name=model_name,
                    run_block=run_block,
                )
            else:
                write_lammps_input(
                    task_dir=task_dir,
                    symbols=symbols,
                    relax=relax,
                    model_name=model_name,
                )

            in_lammps_file_paths.append(Path(task_dir / "in.lammps"))
            lammps_data_file_paths.append(Path(task_dir / "lammps.data"))
            task_paths.append(Path(task_dir))
            task_infos.append({
                "task_name": task_name,
                "strain": strain,
                "lammps_run_method": lammps_run_method,
                "ensemble": selected_ensemble,
                "temp": temp,
                "pres": pres,
                "axis": axis,
                "reference_length": L0,
            })
            print(f"Generated LAMMPS task: {task_name}")

    print("All LAMMPS tasks generated.")
    return {
        "in_lammps_file_paths": in_lammps_file_paths,
        "lammps_data_file_paths": lammps_data_file_paths,
        "task_paths": task_paths,
        "task_infos": task_infos,
    }
