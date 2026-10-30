from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class BuildSupercellResult(TypedDict):
    stacked_system_paths: List[Path]
    system_infos: List[Dict[str, Any]]


class PrepLammpsTasksResult(TypedDict):
    task_root_path: Path
    task_paths: List[Path]
    task_names: List[str]
    task_infos: List[Dict[str, Any]]


class RunDpnegfLammpsResult(TypedDict):
    log_path: Path
    relaxed_system_archive_path: Path
    extra_outputs_path: Optional[Path]
    task_name: str


class PrepNegfTasksResult(TypedDict):
    task_names: List[str]
    modified_negf_input_configs: List[Dict[str, Any]]


class RunDpnegfResult(TypedDict):
    log_paths: List[Path]
    negf_result_paths: List[Path]
    extra_output_archives: List[Path]
    task_name: str


class TransmissionPlotResult(TypedDict):
    transmission_plot_path: str


class AbacusOverlapResult(TypedDict):
    stru_path: Path
    input_path: Path
    running_log_path: Path
    overlap_csr_path: Path


class ConvertOverlapResult(TypedDict):
    overlap_h5_path: Path
