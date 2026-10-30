from pathlib import Path
from typing import List, Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.abacus.core import (
    _abacus_band_gap,
    _abacus_band_plot,
    _abacus_get_fermi_level,
    _run_abacus,
    _run_abacus_scf_nscf,
)
from dptb_pilot.tools.modules.abacus.results import (
    AbacusRunResult,
    BandGapResult,
    BandPlotResult,
    FermiLevelResult,
)


@mcp.tool()
def run_abacus(
    stru_path: Path,
    input_path: Path,
    kpt_path: Optional[Path] = None,
    command: str = "abacus",
    pp_orb_paths: Optional[List[Path]] = None,
    other_file_path: Optional[Path] = None,
) -> AbacusRunResult:
    """Run a single ABACUS calculation and return its complete working directory.

    Parameters
    ----------
    stru_path : Path
        ABACUS ``STRU`` file.
    input_path : Path
        ABACUS ``INPUT`` file.
    kpt_path : Path, optional
        ABACUS ``KPT`` file. Omit it when the input uses ``kspacing``.
    command : str, optional
        ABACUS executable command, including an optional MPI launcher.
    pp_orb_paths : List[Path], optional
        Pseudopotential and numerical-orbital files copied into the run directory.
    other_file_path : Path, optional
        Additional file or directory copied into the run directory.

    Returns
    -------
    AbacusRunResult
        ``abacus_output_path`` points to the working directory containing
        ``INPUT``, ``STRU``, optional ``KPT``, supporting files, and
        ``OUT.ABACUS``. Pass it directly to ``dftio_parse(work_root=...)``.
    """
    return _run_abacus(
        stru_path, input_path, kpt_path, command, pp_orb_paths, other_file_path
    )


@mcp.tool()
def run_abacus_scf_nscf(
    stru_path: Path,
    scf_input_path: Path,
    nscf_input_path: Path,
    nscf_kpt_path: Path,
    command: str = "abacus",
    pp_orb_paths: Optional[List[Path]] = None,
    other_file_path: Optional[Path] = None,
) -> AbacusRunResult:
    """Run ABACUS SCF with kspacing, then NSCF with a K-line path.

    Parameters
    ----------
    stru_path : Path
        ABACUS ``STRU`` file shared by both stages.
    scf_input_path : Path
        SCF ``INPUT`` file. It should define ``kspacing`` and write charge
        density, for example with ``out_chg 1``.
    nscf_input_path : Path
        NSCF ``INPUT`` file. It should use ``init_chg file`` and point
        ``read_file_dir`` at ``./OUT.ABACUS/``.
    nscf_kpt_path : Path
        Line-mode ``KPT`` file installed only after SCF completes.
    command : str, optional
        ABACUS command, including an optional MPI launcher.
    pp_orb_paths : List[Path], optional
        Pseudopotential and orbital files copied into the run directory.
    other_file_path : Path, optional
        Extra file or directory copied into the run directory.

    Returns
    -------
    AbacusRunResult
        ``abacus_output_path`` points to the complete working directory,
        including the final NSCF ``INPUT`` and ``KPT`` and ``OUT.ABACUS``.
        Pass it directly to ``dftio_parse(work_root=...)``.
    """
    return _run_abacus_scf_nscf(
        stru_path,
        scf_input_path,
        nscf_input_path,
        nscf_kpt_path,
        command,
        pp_orb_paths,
        other_file_path,
    )


@mcp.tool()
def abacus_get_fermi_level(abacus_out_path: Path) -> FermiLevelResult:
    """Extract the Fermi level from an ABACUS output directory.

    Parameters
    ----------
    abacus_out_path : Path
        ABACUS working directory or its ``OUT.ABACUS`` subdirectory containing
        ``running_nscf.log`` or another ``running_*.log`` file.

    Returns
    -------
    FermiLevelResult
        Extracted Fermi energy in eV.
    """
    return _abacus_get_fermi_level(abacus_out_path)


@mcp.tool()
def abacus_band_plot(
    abacus_out_path: Path, emin: float = None, emax: float = None
) -> BandPlotResult:
    """Plot ``BANDS_1.dat`` relative to the ABACUS Fermi level.

    Parameters
    ----------
    abacus_out_path : Path
        ABACUS working directory or its ``OUT.ABACUS`` subdirectory containing
        band data and a running log.
    emin : float, optional
        Lower y-axis bound in eV relative to the Fermi level.
    emax : float, optional
        Upper y-axis bound in eV relative to the Fermi level.

    Returns
    -------
    BandPlotResult
        Path to the generated band image.
    """
    return _abacus_band_plot(abacus_out_path, emin, emax)


@mcp.tool()
def abacus_band_gap(
    abacus_out_path: Path,
    use_smart_band_gap: bool = False,
    smart_band_gap_atol: float = 1e-3,
    smart_band_gap_max_diff: float = 1.0,
) -> BandGapResult:
    """Calculate an ABACUS gap, optionally using DeePTB's smart gap search.

    ``BANDS_1.dat`` is converted from ABACUS's text columns into the
    ``(Nk, Nband)`` eigenvalue array expected by DeePTB. Smart-gap VBM and CBM
    are returned relative to the ABACUS Fermi level; the additional
    ``gap_lower`` and ``gap_upper`` fields retain absolute energies.

    Parameters
    ----------
    abacus_out_path : Path
        ABACUS working directory or its ``OUT.ABACUS`` subdirectory containing
        ``BANDS_1.dat`` and a running log.
    use_smart_band_gap : bool, optional
        Use DeePTB's near-Fermi smart/local gap search instead of the
        conventional occupied/unoccupied split.
    smart_band_gap_atol : float, optional
        Minimum resolved gap for smart search in eV.
    smart_band_gap_max_diff : float, optional
        Smart-search radius around the Fermi level in eV.

    Returns
    -------
    BandGapResult
        Band-gap edges, width, and smart-search diagnostics when requested.
    """
    return _abacus_band_gap(
        abacus_out_path,
        use_smart_band_gap,
        smart_band_gap_atol,
        smart_band_gap_max_diff,
    )
