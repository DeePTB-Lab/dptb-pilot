import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write


def _direction_to_index_and_matrix(direction: str) -> Tuple[int, np.ndarray]:
    if direction == "x":
        return 0, np.array([1, 0, 0])
    if direction == "y":
        return 1, np.array([0, 1, 0])
    if direction == "z":
        return 2, np.array([0, 0, 1])
    raise TypeError(f"direction {direction} is not legal!")


def _sort_atoms_in_direction(system: Atoms, direction_index: int) -> Atoms:
    directional_coords = system.positions[:, direction_index]
    sorted_indices = np.argsort(directional_coords)
    return Atoms(
        symbols=system.symbols[sorted_indices],
        positions=system.positions[sorted_indices],
        cell=system.cell,
        pbc=system.pbc,
    )


def _stack_system(init_system: Atoms, output_file: Path, negf_config: Dict[str, Any]):
    direction_index, supercell_matrix = _direction_to_index_and_matrix(negf_config["direction"])
    sorted_system = _sort_atoms_in_direction(init_system, direction_index)

    if "auto_scale_length" in negf_config and negf_config["auto_scale_length"] is not None:
        structure_length = sorted_system.cell.lengths()[direction_index]
        mult = int(negf_config["auto_scale_length"] // structure_length) + 1
        sorted_system = sorted_system.repeat((1, 1, 1) + (mult - 1) * supercell_matrix)
        sorted_system = _sort_atoms_in_direction(sorted_system, direction_index)
    else:
        mult = 1

    repeat = sum(negf_config["supercell"].values())
    supercell = sorted_system.repeat((1, 1, 1) + (repeat - 1) * supercell_matrix)

    pos = supercell.get_positions()
    n_cell = int(negf_config["supercell"]["lead_L"] / 2)
    atom_number_of_layer = len(sorted_system) * n_cell
    cell_length = sorted_system.cell[direction_index, direction_index]
    new_pos = np.vstack([
        pos[:atom_number_of_layer] + supercell_matrix * cell_length * n_cell,
        pos[atom_number_of_layer:2 * atom_number_of_layer] - supercell_matrix * cell_length * n_cell,
        pos[2 * atom_number_of_layer:],
    ])
    supercell.set_positions(new_pos)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write(output_file, supercell, format="vasp")
    return (
        negf_config["supercell"]["lead_L"],
        negf_config["supercell"]["device"],
        negf_config["supercell"]["lead_R"],
        mult,
    )


def build_supercell(
        init_conf_paths: List[Path],
        negf_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Build NEGF stacked supercells from input VASP structures."""
    if negf_config["supercell"]["lead_L"] != negf_config["supercell"]["lead_R"]:
        raise AssertionError("lead_L should be equal to lead_R for symmetric leads.")
    if negf_config["supercell"]["lead_L"] % 2 != 0:
        raise AssertionError("lead should be in even number as double principal layers.")

    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    out_systems: List[Path] = []
    system_infos: List[Dict[str, Any]] = []

    for conf in init_conf_paths:
        conf = Path(conf)
        if not conf.exists():
            raise FileNotFoundError(f"Initial configuration not found: {conf}")
        system = read(conf)
        output_file = work_dir / f"stacked_{os.path.basename(conf)}"
        lead_l, device, lead_r, mult = _stack_system(system, output_file, negf_config)
        out_systems.append(output_file.absolute())
        atom_number = len(system) * mult
        system_infos.append({
            "atom_number": atom_number,
            "atom_index": list(np.array([lead_l, lead_l + device, lead_l + device + lead_r]) * atom_number),
        })

    return {"stacked_system_paths": out_systems, "system_infos": system_infos}
