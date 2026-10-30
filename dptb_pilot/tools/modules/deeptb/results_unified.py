from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class ModelResult(TypedDict):
    model_path: Path


class BandResult(TypedDict):
    band_structure_file_path: Path
    image_file_path: Path
    fermi_level: float


class DosResult(TypedDict):
    dos_file_path: Path
    image_file_path: Path
    fermi_level: float


class DosPlotResult(TypedDict):
    dos_plot_path: Path


class HamiltonianResult(TypedDict):
    hamiltonian_file_path: Path
    overlap_file_path: Path


class RunNegfResult(TypedDict):
    negf_result_file_path: Path
    log_file_path: Path


class RunLammpsResult(TypedDict):
    relaxed_system_file_path: Path


class ConfigResult(TypedDict):
    config_path: Path


class AbacusRunResult(TypedDict):
    """Complete calculation directory, including inputs and OUT.ABACUS."""

    abacus_output_path: Path


class FermiLevelResult(TypedDict):
    fermi_level: float


class BandPlotResult(TypedDict):
    abacus_output_path: Path


class BandGapResult(TypedDict):
    band_gap: float
    vbm: Optional[float]
    cbm: Optional[float]
    vbm_band_index: Optional[int]
    cbm_band_index: Optional[int]
    vbm_kpoint_index: Optional[int]
    cbm_kpoint_index: Optional[int]
    gap_lower: Optional[float]
    gap_upper: Optional[float]
    gap_width: float
    gap_center: Optional[float]
    fermi_level: float
    fermi_level_source: str
    calculation_method: str
    distance_to_fermi: Optional[float]
    fermi_inside_gap: Optional[bool]
    all_gaps: List[Dict[str, float]]
    risky: bool
    is_valid: bool
    status: str
    diagnostic_message: Optional[str]
    k_target: Optional[int]
    energy_window_min: Optional[float]
    energy_window_max: Optional[float]


class BandCompareResult(TypedDict):
    band_compare_path: Path


class StructureConvertResult(TypedDict):
    structure_file_path: Path


class DftioParseResult(TypedDict, total=False):
    output_path: Path
    input_mapping: Dict[str, str]


class DeePTBTrainingResult(TypedDict, total=False):
    output_path: Path
    config_path: Path
    log_path: Path
    command: str
    model_path: Path


class HamiltonianTestResult(TypedDict):
    stats: Dict[str, Any]


class PressTubeTaskResult(TypedDict):
    task_root_path: Path
    task_count: int
    task_paths: List[Path]
