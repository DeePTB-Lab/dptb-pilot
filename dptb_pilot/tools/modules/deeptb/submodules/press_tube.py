from pathlib import Path

import numpy as np
from ase.io import read

from dptb_pilot.tools.modules.deeptb.submodules.lammps import write_lammps_data, generate_group_lines_by_ranges
from dptb_pilot.tools.modules.deeptb.submodules.supercell import build_supercell, make_cylinder_indenter


def build_and_generate(poscar_path: str,
                       out_root: str = "tasks_indents",
                       target_length: float = 100.0,
                       axis='auto',
                       n_repeat=None,
                       indenter_radius=4.0,
                       indenter_height_factor=1.2,
                       indent_depths=None,
                       indenter_spacing=1.42,
                       indenter_symbol='C',
                       deepmd_model_name="MODEL.pb",
                       ensemble='min',
                       dt=1.0,
                       nsteps=5000):
    """
    poscar_path: POSCAR 或 VASP 结构路径 (ASE 可读)
    target_length: 希望的轴向长度 (Å)
    indent_depths: list of lateral offsets (Å) 表示 indenter 距离中心轴的横向距离；值越小表示越深（靠近管子）
    indenter_radius: 柱半径 (Å)
    indenter_height_factor: indenter 高度 = cell_axis_length * factor
    """
    poscar_path = Path(poscar_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    system = read(str(poscar_path))
    # build supercell
    supercell, axis_idx, rep = build_supercell(system, axis=axis, target_length=target_length, n_repeat=n_repeat)

    # center supercell positions within box (optional)
    # ensure periodic box is consistent
    cell = supercell.get_cell()
    lengths = [np.linalg.norm(cell[i]) for i in range(3)]
    axis_length = lengths[axis_idx]

    # compute indenter height
    indenter_height = axis_length * indenter_height_factor

    # determine center of box in x,y,z
    # we place cylinder axis perpendicular to tube axis, so choose center along tube mid z
    com = supercell.get_positions().mean(axis=0)
    cx = 0.5 * np.linalg.norm(cell[0])  # default box center x (we'll use box coords)
    cy = 0.5 * np.linalg.norm(cell[1])
    cz = 0.0  # for cylinder base we will use min z - small margin
    # get bounding box
    positions = supercell.get_positions()
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    # cylinder center in z: choose bottom z slightly below z_min so cylinder covers tube axis
    base_z = z_min - 0.5  # Å margin

    # Setup default indent depths if None:
    if indent_depths is None:
        # produce a set of lateral distances from  (cx + offset)
        # compute approximate tube radial extent in x direction
        tube_radius_x = max(abs(positions[:, 0] - 0.5 * (x_min + x_max)))
        # choose offsets from outside to near center
        indent_depths = list(np.linspace(tube_radius_x + indenter_radius + 3.0, max(tube_radius_x - 1.0, 0.5), num=5))
        # ensure descending (from far -> near)
        indent_depths = sorted(indent_depths, reverse=True)

    # For writing lammps data we will append indenter atoms to the supercell copy
    for i, lateral in enumerate(indent_depths):
        task_name = f"indent_{i:02d}_lat{lateral:.2f}A"
        task_dir = out_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # copy supercell
        atoms_copy = supercell.copy()

        # choose cylinder center coordinates:
        # put cylinder center at (cx_pos, cy_pos, base_z)
        # place cylinder axis parallel to axis vector -> we assume axis_idx is 2 (z) common case
        # we will put cylinder axis along y if axis_idx==2? to keep generality:
        # we choose coordinates such that cylinder axis is perpendicular to tube axis:
        if axis_idx == 2:
            # tube axis along z; cylinder axis along y; so cylinder center x = center_x +/- lateral
            cx_pos = 0.5 * (x_min + x_max) + lateral
            cy_pos = 0.5 * (y_min + y_max)
            cz_pos = base_z
        elif axis_idx == 1:
            cx_pos = 0.5 * (x_min + x_max)
            cy_pos = 0.5 * (y_min + y_max) + lateral
            cz_pos = base_z
        else:
            # axis_idx == 0
            cx_pos = 0.5 * (x_min + x_max)
            cy_pos = 0.5 * (y_min + y_max)
            cz_pos = 0.5 * (z_min + z_max) + lateral

        ind = make_cylinder_indenter(center=(cx_pos, cy_pos, cz_pos),
                                     radius=indenter_radius,
                                     height=indenter_height,
                                     spacing=indenter_spacing,
                                     symbol=indenter_symbol)
        # translate indenter slightly to avoid overlap
        # place it so its bottom base at cz_pos and cylinder extends +z
        # concatenate atoms
        if len(ind) == 0:
            print(f"[warn] indenter empty at lateral={lateral:.2f} Å, skipping")
            continue

        # append indenter atoms to atoms_copy
        atoms_all = atoms_copy + ind

        # write lammps.data
        data_path = task_dir / "lammps.data"
        # we pick specorder so that 'C' appears once (all C)
        specorder = None
        write_lammps_data(atoms_all, str(data_path), specorder=specorder)

        # find indices for indenter atoms (they are the last N atoms)
        N_total = len(atoms_all)
        N_ind = len(ind)
        indenter_ids = list(range(N_total - N_ind + 1, N_total + 1))  # 1-based indices for LAMMPS
        # optionally pick fixed atoms near ends: choose atoms with z < z_min + margin or z > z_max - margin
        positions_all = atoms_all.get_positions()
        fixed_ids = []
        # fix atoms in the first and last ~5Å of cell along axis
        margin = min((axis_length * 0.05), 5.0)
        for idx, pos in enumerate(positions_all):
            coord = pos[axis_idx]
            if coord < (z_min + margin) or coord > (z_max - margin):
                fixed_ids.append(idx + 1)

        group_lines = generate_group_lines_by_ranges(mobile_count=len(atoms_all), fixed_ids=fixed_ids,
                                                     indenter_ids=indenter_ids)

        # create in.lammps content
        in_path = task_dir / "in.lammps"
        with open(in_path, "w", encoding="utf-8") as fh:
            # mass lines: simple mapping for elements present
            unique_syms = []
            for s in atoms_all.get_chemical_symbols():
                if s not in unique_syms:
                    unique_syms.append(s)
            # mass lines
            mass_lines = []
            from ase.data import atomic_numbers, atomic_masses
            for i_sym, sym in enumerate(unique_syms, start=1):
                z = atomic_numbers[sym]
                mass_lines.append(f"mass {i_sym} {float(atomic_masses[z]):.6f}  # {sym}")
            mass_block = "\n".join(mass_lines)

            # pair coeff placeholder (user must adjust model path)
            # build specorder string for pair_coeff: assume one element only -> use symbol repeated
            # If multiple species exist, the user should tailor pair_coeff
            pair_coeff_line = f"pair_style      deepmd {deepmd_model_name}\npair_coeff      * * {' '.join(unique_syms)}\n"

            # ensemble block
            ensemble_block = ""
            if ensemble.lower() == "min":
                ensemble_block = (
                    "thermo          100\n"
                    "min_style       cg\n"
                    f"minimize        1e-6 1e-8 1000 {nsteps}\n"
                )
            elif ensemble.lower() == "nvt":
                ensemble_block = (
                    f"velocity        mobile create 300 12345 mom yes rot yes dist gaussian\n"
                    f"fix             1 mobile nvt temp 300 300 0.1\n"
                    "thermo          100\n"
                    f"timestep        {dt}\n"
                    f"run             {nsteps}\n"
                )
            else:
                ensemble_block = (
                    f"velocity        mobile create 300 12345 mom yes rot yes dist gaussian\n"
                    f"fix             1 mobile nve\n"
                    "thermo          100\n"
                    f"timestep        {dt}\n"
                    f"run             {nsteps}\n"
                )

            # write file
            fh.write(f"""# Auto-generated in.lammps for lateral indent {lateral:.3f} Å
units           metal
atom_style      atomic
boundary        p p p

read_data       lammps.data

{mass_block}

{pair_coeff_line}

neighbor        2.0 bin
neigh_modify    every 10 delay 0 check yes

# groups (fixed/mobile/indenter)
{group_lines}

# keep fixed atoms fixed
fix hold fixed setforce 0.0 0.0 0.0

# keep indenter rigid by removing forces; we do not move it here (static indentation)
fix indenter_fix indenter setforce 0.0 0.0 0.0

# relax mobile atoms (minimization or short MD)
{ensemble_block}

# write out relaxed structure
write_data      relaxed.data
""")
        print(f"[out] wrote task '{task_name}' -> {task_dir}")

    print("[done] generation finished.")

"""
# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build CNT supercell from POSCAR and generate LAMMPS indentation cases")
    p.add_argument("poscar", help="POSCAR or structure file")
    p.add_argument("--out", default="tasks_indents", help="output root dir")
    p.add_argument("--length", type=float, default=100.0, help="target axial length (Å)")
    p.add_argument("--axis", default='auto', help="axis index 0/1/2 or 'auto'")
    p.add_argument("--nrep", type=int, default=None, help="explicit repeat count (overrides length)")
    p.add_argument("--indrad", type=float, default=4.0, help="indenter radius (Å)")
    p.add_argument("--depths", type=str, default=None, help="comma separated lateral offsets Å (e.g. 10.0,8.0,6.0)")
    p.add_argument("--model", default="MODEL.pb", help="deepmd model filename placeholder")
    p.add_argument("--ensemble", default="min", help="min/nvt/nve")
    args = p.parse_args()
    if args.depths:
        depths = [float(x) for x in args.depths.split(",")]
    else:
        depths = None
    return args, depths


if __name__ == "__main__":
    args, depths = parse_args()
    build_and_generate(poscar_path=args.poscar,
                       out_root=args.out,
                       target_length=args.length,
                       axis=args.axis,
                       n_repeat=args.nrep,
                       indenter_radius=args.indrad,
                       indent_depths=depths,
                       deepmd_model_name=args.model,
                       ensemble=args.ensemble)
"""