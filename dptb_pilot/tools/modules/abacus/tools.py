from pathlib import Path
from typing import List, Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.abacus.core import (
    _abacus_band_gap,
    _abacus_band_plot,
    _abacus_get_efermi,
    _run_abacus,
    _run_abacus_scf_nscf,
)
from dptb_pilot.tools.modules.abacus.results import (
    AbacusRunResult,
    BandGapResult,
    BandPlotResult,
    EfermiResult,
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
    """Run a single ABACUS calculation and collect ``OUT.ABACUS``."""
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
def abacus_get_efermi(abacus_out_path: Path) -> EfermiResult:
    """Extract the Fermi level from an ABACUS output directory."""
    return _abacus_get_efermi(abacus_out_path)


@mcp.tool()
def abacus_band_plot(
    abacus_out_path: Path, emin: float = None, emax: float = None
) -> BandPlotResult:
    """Plot ``BANDS_1.dat`` relative to the ABACUS Fermi level."""
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
    """
    return _abacus_band_gap(
        abacus_out_path,
        use_smart_band_gap,
        smart_band_gap_atol,
        smart_band_gap_max_diff,
    )
