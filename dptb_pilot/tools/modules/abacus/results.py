from pathlib import Path
from typing import Any, List, TypedDict


class AbacusRunResult(TypedDict):
    abacus_output_path: Path


class EfermiResult(TypedDict):
    efermi: float


class BandPlotResult(TypedDict):
    abacus_output_path: Path


class BandGapResult(TypedDict, total=False):
    band_gap: float
    vbm: float
    cbm: float
    gap_lower: float
    gap_upper: float
    gap_width: float
    gap_center: float
    distance_to_fermi: float
    fermi_inside_gap: bool
    all_gaps: List[Any]
    risky: bool
