from pathlib import Path
from typing import Sequence, Union

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import DftioParseResult
from dptb_pilot.tools.modules.deeptb.submodules.dftio import _dftio_parse


@mcp.tool()
def dftio_parse(
        work_root: Union[Path, str, Sequence[Union[Path, str]]],
        mode: str = "abacus",
        prefix: Union[str, Sequence[str]] = "abacus",
        output_dir_name: str = "parse_result",
        out_hamiltonian: bool = False,
        out_overlap: bool = False,
        out_density_matrix: bool = False,
        out_eigenvalue: bool = False
) -> DftioParseResult:
    """
    Parse raw DFT outputs into DeePTB/dftio dataset files.

    Parameters
    ----------
    work_root : Path, str, or sequence of paths
        A dataset root, one calculation directory, or multiple calculation
        directories. Inputs are normalized automatically before parsing.
    mode : str, optional
        Parser backend passed to ``dftio parse``, usually ``abacus``.
    prefix : str or sequence of str, optional
        Optional selection hint when ``work_root`` is a dataset root. It is
        never passed through as a path, so prefixes containing separators are
        handled safely.
    output_dir_name : str, optional
        Name of the output directory copied into the generated work path.
    out_hamiltonian : bool, optional
        Export Hamiltonian matrices.
    out_overlap : bool, optional
        Export overlap matrices.
    out_density_matrix : bool, optional
        Export density matrices.
    out_eigenvalue : bool, optional
        Export eigenvalues.

    Returns
    -------
    DftioParseResult
        Dictionary with ``output_path`` and an ``input_mapping`` from each
        resolved source directory to its parsed output directory.
    """
    return _dftio_parse(
        work_root=work_root,
        mode=mode,
        prefix=prefix,
        output_dir_name=output_dir_name,
        out_hamiltonian=out_hamiltonian,
        out_overlap=out_overlap,
        out_density_matrix=out_density_matrix,
        out_eigenvalue=out_eigenvalue,
    )
