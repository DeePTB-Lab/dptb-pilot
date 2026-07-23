from pathlib import Path
from typing import NotRequired, TypedDict


class DeepMDTrainingResult(TypedDict):
    output_path: Path
    log_path: Path
    command: str
    model_path: NotRequired[Path]
    pretrained_model_path: NotRequired[Path]
