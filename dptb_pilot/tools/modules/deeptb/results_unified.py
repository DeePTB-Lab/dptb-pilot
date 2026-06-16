from pathlib import Path
from typing import Any, Dict, List, TypedDict


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
    abacus_output_path: Path


class EfermiResult(TypedDict):
    efermi: float


class BandPlotResult(TypedDict):
    abacus_output_path: Path


class BandGapResult(TypedDict, total=False):
    band_gap: float
    vbm: float
    cbm: float
    vbm_band_index: int
    cbm_band_index: int


class BandCompareResult(TypedDict):
    band_compare_path: Path


class StructureConvertResult(TypedDict):
    structure_file_path: Path


class DftioParseResult(TypedDict):
    output_path: Path


class HamiltonianTestResult(TypedDict):
    stats: Dict[str, Any]


class PressTubeTaskResult(TypedDict):
    task_root_path: Path
    task_count: int
    task_paths: List[Path]
