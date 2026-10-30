import os
from pathlib import Path
from typing import List, Optional, Union

import dpdata
from ase.io import read, write


def convert_from_lammps_data(
    input: Path,
    output_name: str = "auto",
    output_path: Path = None,
    format: str = "vasp",
    pp_file_paths: Optional[List[Path]] = None,
    orb_file_paths: Optional[List[Path]] = None
) -> Path:
    """
    Convert LAMMPS data file to VASP (POSCAR) or ABACUS (STRU).

    Parameters
    ----------
    input : Path
        Input LAMMPS data file.
    output_name : str
        Output file name. If "auto", use default naming.
    format : str
        Target format: "vasp" or "abacus".

    Returns
    -------
    Path
        Path to written structure file.
    """
    if pp_file_paths:
        if isinstance(pp_file_paths[0], Path):
            pp_file_paths = [str(i.absolute()) for i in pp_file_paths]
    if orb_file_paths:
        if isinstance(orb_file_paths[0], Path):
            orb_file_paths = [str(i.absolute()) for i in orb_file_paths]

    input = Path(input).resolve()

    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if format not in ("vasp", "abacus"):
        raise ValueError("format must be 'vasp' or 'abacus'")

    # 读取 LAMMPS data 文件
    atoms = read(input, format="lammps-data")

    # ===== 自动命名逻辑 =====
    if output_name == "auto":
        if format == "vasp":
            filename = "POSCAR"
        elif format == "abacus":
            filename = "STRU"
    else:
        filename = output_name

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_path = output_path / filename
    else:
        output_path = input.parent / filename

    # ===== 避免覆盖 =====
    counter = 1
    base = output_path
    while output_path.exists():
        output_path = base.with_name(f"{base.name}_{counter}")
        counter += 1

    # ===== 写出 =====
    if format == "vasp":
        write(output_path, atoms, format='vasp')
    if format == "abacus":
        write(input.parent / 'POSCAR', atoms, format='vasp')
        (dpdata.System(input.parent / 'POSCAR', fmt="vasp/poscar")
         .to("abacus/stru", output_path,
             pp_file=pp_file_paths,
             numerical_orbital=orb_file_paths))

    print(f"Written {format.upper()} file: {output_path}")

    return output_path

def convert_from_vasp_poscar(
    input: Path,
    output_name: str = "auto",
    output_path: Path = None,
    format: str = "abacus",
    pp_file_paths: Union[List[Path], List[str], None] = None,
    orb_file_paths: Union[List[Path], List[str], None] = None,
) -> Path:
    """
    Convert LAMMPS data file to VASP (POSCAR) or ABACUS (STRU).

    Parameters
    ----------
    input : Path
        Input LAMMPS data file.
    output_name : str
        Output file name. If "auto", use default naming.
    format : str
        Target format: "vasp" or "abacus".

    Returns
    -------
    Path
        Path to written structure file.
    """
    if pp_file_paths:
        if isinstance(pp_file_paths[0], Path):
            pp_file_paths = [str(i.absolute()) for i in pp_file_paths]
    if orb_file_paths:
        if isinstance(orb_file_paths[0], Path):
            orb_file_paths = [str(i.absolute()) for i in orb_file_paths]

    input = Path(input).resolve()

    if not input.exists():
        raise FileNotFoundError(f"Input file not found: {input}")

    if format not in ("vasp", "abacus"):
        raise ValueError("format must be 'vasp' or 'abacus'")

    # 读取 LAMMPS data 文件
    atoms = read(input, format="vasp")

    # ===== 自动命名逻辑 =====
    if output_name == "auto":
        if format == "vasp":
            filename = "POSCAR"
        elif format == "abacus":
            filename = "STRU"
    else:
        filename = output_name

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_path = output_path / filename
    else:
        output_path = input.parent / filename

    # ===== 避免覆盖 =====
    counter = 1
    base = output_path
    while output_path.exists():
        output_path = base.with_name(f"{base.name}_{counter}")
        counter += 1

    # ===== 写出 =====
    if format == "vasp":
        write(output_path, atoms, format='vasp')
    if format == "abacus":
        write(input.parent / 'POSCAR', atoms, format='vasp')
        (dpdata.System(input.parent / 'POSCAR', fmt="vasp/poscar")
         .to("abacus/stru", output_path,
             pp_file=pp_file_paths,
             numerical_orbital=orb_file_paths))

    print(f"Written {format.upper()} file: {output_path}")

    return output_path