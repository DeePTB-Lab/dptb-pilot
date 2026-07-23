import shutil
import tempfile
from pathlib import Path
import subprocess as sp
import tempfile
import os
import shutil
from dftio.io.abacus.abacus_parser import AbacusParser
from tqdm import tqdm

from dptb_pilot.tools.modules.util.comm import generate_work_path

def _dftio_parse_abacus(
        work_root: Path,
        prefix: str = "abacus",
        format: str = "dat",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
    ):
    """
    Parse ABACUS raw calculation folders with the Python ``dftio`` parser.

    Parameters
    ----------
    work_root : Path
        Root directory that contains ABACUS calculation data.
    prefix : str, optional
        Dataset folder prefix used by ``AbacusParser``.
    output_dir_name : str, optional
        Name of the parser output directory.
    out_hamiltonian, out_overlap, out_density_matrix, out_eigenvalue : bool
        Select which quantities should be written to HDF5 files.

    Notes
    -----
    This helper writes parser outputs through ``AbacusParser.write_hdf5`` and does
    not currently return the generated paths.
    """
    work_root = work_root.absolute()

    parser = AbacusParser(
        root=str(work_root),
        prefix=prefix,
    )
    num_entries = len(parser.raw_datas)

    for idx in tqdm(range(num_entries)):
        parser.write(
            idx=int(idx),
            format=format,
            hamiltonian=out_hamiltonian,
            overlap=out_overlap,
            outroot=work_root / output_dir_name,
            eigenvalue=out_eigenvalue,
            density_matrix=out_density_matrix,
            band_index_min=0
        )

def _dftio_parse(
        work_root: Path,
        mode: str = "abacus",
        prefix: str = "abacus",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
    ):
    """
    Convert DFT output folders into DeePTB/dftio training data with the ``dftio`` CLI.

    Parameters
    ----------
    work_root : Path
        Root directory containing raw DFT calculation folders.
    mode : str, optional
        Parser backend passed to ``dftio parse``; currently typically ``abacus``.
    prefix : str, optional
        Prefix of calculation folders to parse.
    output_dir_name : str, optional
        Name of the copied output directory under the generated work path.
    out_hamiltonian, out_overlap, out_density_matrix, out_eigenvalue : bool
        Flags controlling which physical quantities are exported.

    Returns
    -------
    dict
        Dictionary with ``output_path`` pointing to the parse result directory.

    Raises
    ------
    RuntimeError
        If the ``dftio parse`` command exits with a non-zero status.
    """
    work_path = Path(generate_work_path()).absolute()

    assert work_root, "根路径必须输入"

    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)

        cmd = ["dftio", "parse",
               "-m", mode,
               "-r", work_root,
               "-p", prefix,
               "-o", temp_path]

        if out_hamiltonian:
            cmd.append("-ham")
        if out_overlap:
            cmd.append("-ovp")
        if out_density_matrix:
            cmd.append("-dm")
        if out_eigenvalue:
            cmd.append("-eig")

        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"dftio execution failed:\n{result.stderr}")

        output_path = work_path / output_dir_name
        shutil.copytree(temp_path,
                        output_path,
                        dirs_exist_ok=True)

    return {"output_path": output_path}




