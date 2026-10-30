import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write

from dptb_pilot.tools.modules.dpnegf.submodules.archive import pack_files
from dptb_pilot.tools.modules.util.comm import run_command


def _build_specorder(system: Atoms) -> List[str]:
    spec = []
    for symbol in system.get_chemical_symbols():
        if symbol not in spec:
            spec.append(symbol)
    return spec


def _mass_lines(specorder: List[str]) -> str:
    lines = []
    for i, symbol in enumerate(specorder, start=1):
        z = atomic_numbers[symbol]
        mass = float(atomic_masses[z])
        lines.append(f"mass {i} {mass:.6f}  # {symbol}")
    return "\n".join(lines)


def _group_fixed_by_ids(fixed_ids: List[int]) -> str:
    if not fixed_ids:
        return "# no fixed atoms"
    ids = [str(i) for i in sorted(set(fixed_ids))]
    lines = ["group fixed empty"]
    lines.extend(
        f"group fixed id {' '.join(ids[start:start + 64])}"
        for start in range(0, len(ids), 64)
    )
    lines.extend([
        "group mobile subtract all fixed",
        "velocity fixed set 0.0 0.0 0.0",
        "fix freeze fixed setforce 0.0 0.0 0.0",
    ])
    return "\n".join(lines)


def _ensemble_block(
        ensemble: str,
        temp: float,
        pressure: float,
        dt: float,
        nsteps: int,
        temp_0: float = 50,
        nstep_heating: int = 10000,
        nstep_equil: int = 10000,
        n_out: int = 10,
        no_group: bool = False,
) -> str:
    group_name = "all" if no_group else "mobile"
    ensemble_lower = ensemble.lower()
    if ensemble_lower == "min":
        return (
            "thermo          100\n"
            "min_style       cg\n"
            f"minimize        1e-6 1e-8 1000 {nsteps}"
        )
    if ensemble_lower == "nvt":
        return (
            f"velocity        {group_name} create {temp} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} nvt temp {temp} {temp} 0.1\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    if ensemble_lower == "npt":
        return (
            f"velocity        {group_name} create {temp} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} npt temp {temp} {temp} 0.1 iso {pressure:.6f} {pressure:.6f} 1.0 dilate {group_name}\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    if ensemble_lower == "nve":
        return (
            f"velocity        {group_name} create {temp} 12345 mom yes rot yes dist gaussian\n"
            f"fix             1 {group_name} nve\n"
            "thermo          100\n"
            f"timestep        {dt}\n"
            f"run             {nsteps}"
        )
    if ensemble_lower == "smart":
        return "\n".join([
            "# -----------------------------",
            "# 0. 能量最小化",
            "min_style       cg",
            "minimize        1e-10 1e-10 10000 10000",
            "reset_timestep  0",
            "# -----------------------------",
            "# 1. 初始速度（低温启动）",
            f"velocity {group_name} create {temp_0} 12345 mom yes rot yes dist gaussian",
            "# -----------------------------",
            "# 2. 缓慢升温（50K → 目标温度）",
            f"variable T equal {temp}",
            f"fix heat {group_name} nvt temp {temp_0} {temp} 0.1",
            "thermo 100",
            "timestep 0.001",
            f"run {nstep_heating}     # 50 ps 升温",
            "unfix heat",
            "reset_timestep 0",
            "# -----------------------------",
            "# 3. 恒温弛豫（达到热平衡）",
            f"fix equil {group_name} nvt temp {temp} {temp} 0.1",
            "dump eq all atom 500 equil.xyz",
            f"run {nstep_equil}     # 50 ps 平衡",
            "unfix equil",
            "undump eq",
            "reset_timestep 0",
            "# -----------------------------",
            "# 4. 采样阶段（关键：系综平均）",
            f"fix sample {group_name} nvt temp {temp} {temp} 0.1",
            "# 每1000步输出一个结构（约1 ps间隔）",
            "dump traj all custom 1000 traj.xyz id type xu yu zu",
            'dump_modify traj format line "%d %d %.16f %.16f %.16f"',
            f"run {1000 * (n_out - 1)}    # 100 ps → 得到约100个构型",
            "unfix sample",
            "undump traj",
            "write_data final.data",
        ])
    raise ValueError(f"Unsupported LAMMPS ensemble: {ensemble}")


def prepare_lammps_tasks(
        stacked_system_paths: List[Path],
        system_infos: List[Dict[str, Any]],
        relax_config: Dict[str, Any],
        inputs_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Generate NEGF LAMMPS relaxation task directories."""
    work_dir = Path(work_path).absolute()
    task_root = work_dir / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)

    model_name = os.path.basename(inputs_config["deepmd_model_path"])
    task_paths: List[Path] = []
    task_names: List[str] = []
    task_infos: List[Dict[str, Any]] = []

    for conf, system_info in zip(stacked_system_paths, system_infos):
        system = read(conf)
        fixed_atom_indices = (
            list(range(1, system_info["atom_index"][0] + 1)) +
            list(range(system_info["atom_index"][1] + 1, system_info["atom_index"][2] + 1))
        )

        if "device_end_fixed_radius" in relax_config:
            radius = relax_config["device_end_fixed_radius"]
            cell = system.get_cell()
            cell_z = cell[2][2]
            atom_number = system_info["atom_number"]
            supercell_multiplier = system_info["atom_index"][2] / atom_number
            cell_length = cell_z / supercell_multiplier
            a0, a1 = system_info["atom_index"][0], system_info["atom_index"][1]
            z0_limit = (a0 / atom_number) * cell_length + radius
            z1_limit = (a1 / atom_number) * cell_length - radius
            positions = system.get_positions()
            for idx in range(a0, a1):
                atom_id = idx + 1
                z_pos = positions[idx][2]
                if z_pos < z0_limit or z_pos > z1_limit:
                    fixed_atom_indices.append(atom_id)

        specorder = inputs_config.get("deepmd_model_type_map") or _build_specorder(system)

        for temp in relax_config["temps"]:
            for pressure in relax_config["press"]:
                task_info = {
                    "conf_name": os.path.basename(str(conf)).replace(".vasp", ""),
                    "ensemble": relax_config["ensemble"],
                    "temp": temp,
                    "pres": pressure,
                    "system_info": system_info,
                }
                task_name = f"lmp_relax_{task_info['conf_name']}_{task_info['temp']}K_{task_info['pres']}bar"
                task_dir = task_root / task_name
                task_dir.mkdir(parents=True, exist_ok=True)

                write(task_dir / "lammps.data", system, format="lammps-data", specorder=specorder)
                mass_lines = _mass_lines(specorder)
                group_lines = _group_fixed_by_ids(fixed_atom_indices)
                additional = relax_config.get("additional") or {}
                ensemble_block = _ensemble_block(
                    relax_config["ensemble"],
                    temp,
                    pressure,
                    relax_config["dt"],
                    relax_config["nsteps"],
                    additional.get("temp_0", 50),
                    additional.get("nstep_heating", 10000),
                    additional.get("nstep_equil", 10000),
                    additional.get("n_out", 10),
                    no_group=group_lines == "# no fixed atoms",
                )

                (task_dir / "in.lammps").write_text(f"""# Auto-generated
units           metal
atom_style      atomic
boundary        p p p

read_data       lammps.data

{mass_lines}

pair_style      deepmd {model_name}
pair_coeff      * * {' '.join(specorder)}

neighbor        2.0 bin
neigh_modify    every 10 delay 0 check yes

{group_lines}

{ensemble_block}

write_data      relaxed.data
        """, encoding="utf-8")

                task_paths.append(task_dir.absolute())
                task_names.append(task_name)
                task_infos.append(task_info)

    return {
        "task_root_path": task_root.absolute(),
        "task_paths": task_paths,
        "task_names": task_names,
        "task_infos": task_infos,
    }


def _build_type_to_element_map(data_file: Path) -> Dict[int, str]:
    atoms = read(data_file, format="lammps-data")
    types = atoms.arrays["type"]
    symbols = atoms.get_chemical_symbols()
    type_map = {}
    for atom_type, symbol in zip(types, symbols):
        if atom_type not in type_map:
            type_map[atom_type] = symbol
    return type_map


def _apply_type_map(atoms: Atoms, type_map: Dict[int, str]) -> Atoms:
    if "type" in atoms.arrays:
        types = atoms.arrays["type"]
    elif "types" in atoms.arrays:
        types = atoms.arrays["types"]
    else:
        raise KeyError("未找到 type/types 信息，dump 文件可能格式不对")
    atoms.set_chemical_symbols([type_map[t] for t in types])
    return atoms


def run_lammps_task(
        task_path: Path,
        task_name: str,
        deepmd_model_path: Path,
        relax_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Run one prepared NEGF LAMMPS task and archive relaxed structures."""
    task_path = Path(task_path).absolute()
    deepmd_model_path = Path(deepmd_model_path).absolute()
    work_dir = Path(work_path).absolute() / task_name
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(task_path / "in.lammps", work_dir / "in.lammps")
    shutil.copy(task_path / "lammps.data", work_dir / "lammps.data")
    shutil.copy(deepmd_model_path, work_dir / deepmd_model_path.name)

    command = " ".join([relax_config["run_config"]["command"], "-i", "in.lammps", "-log", "log.lammps"])
    cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        ret, out, err = run_command(command, shell=True)
    finally:
        os.chdir(cwd)
    if ret != 0:
        raise RuntimeError(f"lmp failed\ncommand was: {command}\nout msg: {out}\nerr msg: {err}")

    if relax_config["ensemble"] == "smart":
        type_map = _build_type_to_element_map(work_dir / "lammps.data")
        structures = read(work_dir / "traj.xyz", index=":", format="lammps-dump-text")
        relaxed_system_paths = []
        for i, atoms in enumerate(structures):
            atoms = _apply_type_map(atoms, type_map)
            scaled = atoms.get_scaled_positions(wrap=False)
            scaled[np.abs(scaled[:, 2] - 1.0) < 1e-6, 2] = 1.0 - 1e-8
            scaled[np.abs(scaled[:, 2]) < 1e-6, 2] = 1e-8
            atoms.set_scaled_positions(scaled)
            poscar_name = f"POSCAR_{i:04d}.vasp"
            write(work_dir / poscar_name, atoms, vasp5=True)
            relaxed_system_paths.append(Path(poscar_name))
    else:
        relaxed_system = read(work_dir / "relaxed.data", format="lammps-data")
        write(work_dir / "relaxed.vasp", relaxed_system, vasp5=True)
        relaxed_system_paths = [Path("relaxed.vasp")]

    archive_path = pack_files(work_dir, relaxed_system_paths, "relaxed_system.tar.gz")
    return {
        "log_path": (work_dir / "log.lammps").absolute(),
        "relaxed_system_archive_path": archive_path,
        "extra_outputs_path": None,
        "task_name": task_name,
    }
