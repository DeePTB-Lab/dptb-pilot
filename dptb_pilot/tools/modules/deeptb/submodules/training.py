from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from dptb_pilot.tools.modules.deeptb.results_unified import DeePTBTrainingResult


_DATA_SECTIONS = ("train", "validation", "reference")
_DATA_FILE_REQUIREMENTS = {
    "get_Hamiltonian": (("hamiltonians.h5",),),
    "get_overlap": (("overlaps.h5",),),
    "get_DM": (("density_matrices.h5", "DM.h5"),),
    "get_eigenvalues": (("eigenvalues.npy",), ("kpoints.npy",)),
}


def _load_config(input_config_path: Path) -> dict[str, Any]:
    """Load and minimally validate a DeePTB JSON training configuration."""
    config_path = Path(input_config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DeePTB input config not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError("DeePTB training input must be a JSON file")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid DeePTB JSON config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("DeePTB training input must contain a JSON object")

    data_options = config.get("data_options")
    if not isinstance(data_options, dict) or not isinstance(
        data_options.get("train"), dict
    ):
        raise ValueError("DeePTB config must define data_options.train")
    return config


def _dataset_folders(
    dataset_root: Path,
    section_name: str,
    section_options: Mapping[str, Any],
) -> list[Path]:
    """Find the dftio trajectory folders selected by a data section."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"Dataset for data_options.{section_name} not found: {root}"
        )

    prefix = section_options.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError(f"data_options.{section_name}.prefix must be provided")
    separator = section_options.get("separator", ".")
    if not isinstance(separator, str):
        raise ValueError(f"data_options.{section_name}.separator must be a string")

    folders = sorted(
        path for path in root.glob(f"{prefix}{separator}*") if path.is_dir()
    )
    if not folders:
        raise ValueError(
            f"Dataset {root} has no dftio trajectory folders matching "
            f"{prefix}{separator}* for data_options.{section_name}"
        )
    return folders


def _validate_dataset(
    dataset_root: Path,
    section_name: str,
    section_options: Mapping[str, Any],
) -> None:
    """Verify that a staged dataset matches the data types requested by DeePTB."""
    root = Path(dataset_root).resolve()
    folders = _dataset_folders(root, section_name, section_options)
    requested_types = [
        key for key in _DATA_FILE_REQUIREMENTS if section_options.get(key) is True
    ]
    if not requested_types:
        raise ValueError(
            f"data_options.{section_name} must enable at least one of "
            + ", ".join(_DATA_FILE_REQUIREMENTS)
        )
    if section_options.get("get_Hamiltonian") and section_options.get("get_DM"):
        raise ValueError(
            f"data_options.{section_name} cannot enable get_Hamiltonian and get_DM "
            "at the same time"
        )

    public_info = (root / "info.json").is_file()
    for folder in folders:
        if not public_info and not (folder / "info.json").is_file():
            raise ValueError(
                f"Expected dftio dataset metadata at {root / 'info.json'} or "
                f"{folder / 'info.json'}"
            )

        recognized_data = any(folder.glob("*.dat")) or any(folder.glob("*.traj"))
        recognized_data = recognized_data or any(folder.glob("*.h5")) or any(
            folder.glob("*.mdb")
        )
        if not recognized_data:
            raise ValueError(
                f"Dataset folder does not look like dftio/DeePTB output: {folder}"
            )

        for data_type in requested_types:
            for alternatives in _DATA_FILE_REQUIREMENTS[data_type]:
                if not any((folder / name).is_file() for name in alternatives):
                    expected = " or ".join(alternatives)
                    raise ValueError(
                        f"data_options.{section_name}.{data_type} is true, but "
                        f"{folder} does not contain {expected}"
                    )


def _copy_dataset(source: Path, destination: Path) -> Path:
    """Copy one transferred dataset into the self-contained training run."""
    shutil.copytree(source, destination)
    return destination.resolve()


def _write_log(
    log_path: Path,
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    log_path.write_text(
        f"$ {shlex.join(command)}\n\n[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}",
        encoding="utf-8",
    )


def _latest_checkpoint(output_path: Path) -> Path | None:
    checkpoints = [path for path in output_path.rglob("*.pth") if path.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def train_with_cli(
    input_config_path: Path,
    training_dataset_path: Path,
    validation_dataset_path: Path | None = None,
    reference_dataset_path: Path | None = None,
    work_path: str = ".",
    output_dir_name: str = "deeptb_train",
    dptb_command: str = "dptb",
    init_model_path: Path | None = None,
    restart_model_path: Path | None = None,
    extra_cli_args: list[str] | None = None,
) -> DeePTBTrainingResult:
    """Stage, validate, rewrite, and execute one DeePTB training run."""
    config_source = Path(input_config_path).resolve()
    config = _load_config(config_source)
    if init_model_path is not None and restart_model_path is not None:
        raise ValueError("init_model_path and restart_model_path are mutually exclusive")
    if not output_dir_name or Path(output_dir_name).name != output_dir_name:
        raise ValueError("output_dir_name must be a single directory name")
    command_prefix = shlex.split(dptb_command)
    if not command_prefix:
        raise ValueError("dptb_command cannot be empty")

    checkpoint_sources: list[tuple[str, Path]] = []
    for option, checkpoint_path in (
        ("--init-model", init_model_path),
        ("--restart", restart_model_path),
    ):
        if checkpoint_path is None:
            continue
        checkpoint_source = Path(checkpoint_path).resolve()
        if not checkpoint_source.is_file():
            raise FileNotFoundError(f"DeePTB checkpoint not found: {checkpoint_source}")
        checkpoint_sources.append((option, checkpoint_source))

    train_source = Path(training_dataset_path).resolve()
    dataset_sources = {
        "train": train_source,
        "validation": (
            Path(validation_dataset_path).resolve()
            if validation_dataset_path is not None
            else train_source
        ),
        "reference": (
            Path(reference_dataset_path).resolve()
            if reference_dataset_path is not None
            else train_source
        ),
    }
    data_options = config["data_options"]
    for section_name in _DATA_SECTIONS:
        section_options = data_options.get(section_name)
        if section_options is None:
            continue
        if not isinstance(section_options, dict):
            raise ValueError(f"data_options.{section_name} must be an object")
        _validate_dataset(
            dataset_sources[section_name], section_name, section_options
        )

    output_root = Path(work_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / output_dir_name
    if run_path.exists():
        raise FileExistsError(f"Output directory already exists: {run_path}")
    run_path.mkdir()

    copied_sources: dict[Path, Path] = {}
    for section_name in _DATA_SECTIONS:
        section_options = data_options.get(section_name)
        if section_options is None:
            continue
        source = dataset_sources[section_name]
        staged_root = copied_sources.get(source)
        if staged_root is None:
            staged_root = _copy_dataset(
                source, run_path / "datasets" / section_name
            )
            copied_sources[source] = staged_root
        section_options["root"] = str(staged_root)

    staged_config_path = run_path / config_source.name
    staged_config_path.write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    training_output_path = run_path / "output"
    command = [
        *command_prefix,
        "train",
        staged_config_path.name,
        "-o",
        training_output_path.name,
    ]

    for option, checkpoint_source in checkpoint_sources:
        staged_checkpoint = run_path / "inputs" / checkpoint_source.name
        staged_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_source, staged_checkpoint)
        command.extend([option, str(staged_checkpoint.relative_to(run_path))])

    command.extend(extra_cli_args or [])
    log_path = run_path / "deeptb_train.log"
    try:
        result = subprocess.run(
            command,
            cwd=run_path,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to start DeePTB CLI command {command_prefix[0]!r}. "
            "Install DeePTB in the execution environment or set dptb_command."
        ) from exc
    _write_log(log_path, command, result)
    if result.returncode != 0:
        raise RuntimeError(
            f"DeePTB CLI exited with status {result.returncode}. See {log_path}"
        )
    if not training_output_path.is_dir():
        raise FileNotFoundError(
            f"DeePTB did not create the expected output directory: "
            f"{training_output_path}"
        )

    training_result: DeePTBTrainingResult = {
        "output_path": training_output_path,
        "config_path": staged_config_path,
        "log_path": log_path,
        "command": shlex.join(command),
    }
    model_path = _latest_checkpoint(training_output_path)
    if model_path is not None:
        training_result["model_path"] = model_path
    return training_result
