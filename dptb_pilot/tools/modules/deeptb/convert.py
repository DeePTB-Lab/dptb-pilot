from pathlib import Path
from typing import List, Optional, Union

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import StructureConvertResult
from dptb_pilot.tools.modules.deeptb.submodules.convert import (
    convert_from_lammps_data as _convert_from_lammps_data,
    convert_from_vasp_poscar as _convert_from_vasp_poscar,
)


@mcp.tool()
def convert_lammps_data_structure(
        input: Path,
        output_name: str = "auto",
        output_path: Path = None,
        format: str = "vasp",
        pp_file_paths: Optional[List[Path]] = None,
        orb_file_paths: Optional[List[Path]] = None
) -> StructureConvertResult:
    """
    Convert a LAMMPS data structure file to VASP POSCAR or ABACUS STRU format.

    Parameters
    ----------
    input : Path
        Path to the input LAMMPS data file.
    output_name : str, optional
        Output file name. Use ``auto`` to select ``POSCAR`` or ``STRU``.
    output_path : Path, optional
        Output directory. If omitted, write next to the input file.
    format : str, optional
        Target format: ``vasp`` or ``abacus``.
    pp_file_paths : List[Path], optional
        Pseudopotential files used when writing ABACUS STRU.
    orb_file_paths : List[Path], optional
        Numerical orbital files used when writing ABACUS STRU.

    Returns
    -------
    StructureConvertResult
        Dictionary with ``structure_file_path`` pointing to the generated file.
    """
    return {"structure_file_path": _convert_from_lammps_data(
        input=input,
        output_name=output_name,
        output_path=output_path,
        format=format,
        pp_file_paths=pp_file_paths,
        orb_file_paths=orb_file_paths,
    )}


@mcp.tool()
def convert_vasp_poscar_structure(
        input: Path,
        output_name: str = "auto",
        output_path: Path = None,
        format: str = "abacus",
        pp_file_paths: Union[List[Path], List[str], None] = None,
        orb_file_paths: Union[List[Path], List[str], None] = None,
) -> StructureConvertResult:
    """
    Convert a VASP POSCAR structure to VASP POSCAR or ABACUS STRU format.

    Parameters
    ----------
    input : Path
        Path to the input VASP POSCAR file.
    output_name : str, optional
        Output file name. Use ``auto`` to select ``POSCAR`` or ``STRU``.
    output_path : Path, optional
        Output directory. If omitted, write next to the input file.
    format : str, optional
        Target format: ``vasp`` or ``abacus``.
    pp_file_paths : List[Path] or List[str], optional
        Pseudopotential files used when writing ABACUS STRU.
    orb_file_paths : List[Path] or List[str], optional
        Numerical orbital files used when writing ABACUS STRU.

    Returns
    -------
    StructureConvertResult
        Dictionary with ``structure_file_path`` pointing to the generated file.
    """
    return {"structure_file_path": _convert_from_vasp_poscar(
        input=input,
        output_name=output_name,
        output_path=output_path,
        format=format,
        pp_file_paths=pp_file_paths,
        orb_file_paths=orb_file_paths,
    )}
