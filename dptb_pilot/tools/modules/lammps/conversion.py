from pathlib import Path
from typing import List, Optional, Union

from ase.io import read, write

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.submodules.convert import convert_from_lammps_data


@mcp.tool()
def convert_structure_to_lammps_data(
        structure_file_path: Path,
        output_file_path: Optional[Path] = None,
        element_order: Optional[List[str]] = None,
        index: Union[int, str] = 0,
) -> dict:
    """Convert an ASE-readable structure file to a LAMMPS atomic data file.

    Parameters
    ----------
    structure_file_path : Path
        Input POSCAR, CIF, XYZ, or another ASE-readable structure file.
    output_file_path : Path, optional
        Output path; defaults to ``lammps.data`` next to the input structure.
    element_order : List[str], optional
        Element ordering used to assign LAMMPS atom type numbers. Every element
        in the structure must be included.
    index : int or str, optional
        Structure frame to read from multi-frame formats; ``"-1"`` selects the last.

    Returns
    -------
    dict
        Path to the generated LAMMPS data file and the atom type ordering.
    """
    source = Path(structure_file_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Structure file not found: {source}")
    atoms = read(source, index=index)
    symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    specorder = list(element_order) if element_order else symbols
    if len(set(specorder)) != len(specorder):
        raise ValueError("element_order cannot contain duplicate symbols")
    missing = sorted(set(symbols) - set(specorder))
    if missing:
        raise ValueError(f"element_order does not include structure elements: {missing}")

    destination = Path(output_file_path) if output_file_path else source.with_name("lammps.data")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination, atoms, format="lammps-data", specorder=specorder, atom_style="atomic")
    return {"lammps_data_file_path": destination, "element_order": specorder}


@mcp.tool()
def convert_lammps_data_structure(
        input: Path,
        output_name: str = "auto",
        output_path: Optional[Path] = None,
        format: str = "vasp",
        pp_file_paths: Optional[List[Path]] = None,
        orb_file_paths: Optional[List[Path]] = None,
) -> dict:
    """Convert a LAMMPS data file to VASP POSCAR or ABACUS STRU.

    Parameters
    ----------
    input : Path
        Source LAMMPS data file.
    output_name : str, optional
        Output filename, or ``auto`` for POSCAR/STRU.
    output_path : Path, optional
        Output directory; defaults to the source file's directory.
    format : str, optional
        Target structure format: ``vasp`` or ``abacus``.
    pp_file_paths : List[Path], optional
        ABACUS pseudopotential paths required for STRU conversion.
    orb_file_paths : List[Path], optional
        ABACUS numerical-orbital paths required for STRU conversion.

    Returns
    -------
    dict
        Path to the generated structure file.
    """
    result = convert_from_lammps_data(
        input=Path(input),
        output_name=output_name,
        output_path=Path(output_path) if output_path else None,
        format=format,
        pp_file_paths=pp_file_paths,
        orb_file_paths=orb_file_paths,
    )
    return {"structure_file_path": result}


convert_lammps_data_to_structure = convert_lammps_data_structure
