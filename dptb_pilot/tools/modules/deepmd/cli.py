from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Sequence

from dptb_pilot.tools.modules.deepmd.results import DeepMDTrainingResult

DeepMDBackend = Literal["pt", "jax", "pd", "tf"]


def _resolve_inside(directory: Path, relative_path: str, label: str) -> Path:
    candidate = (directory / relative_path).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the training workspace") from exc
    return candidate


def _stage_workspace(
    training_workspace_path: Path,
    work_path: str,
    output_dir_name: str,
    input_config: str,
) -> tuple[Path, Path]:
    workspace = Path(training_workspace_path).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Training workspace not found: {workspace}")
    if not output_dir_name or Path(output_dir_name).name != output_dir_name:
        raise ValueError("output_dir_name must be a single directory name")

    output_root = Path(work_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / output_dir_name
    try:
        run_dir.relative_to(workspace)
    except ValueError:
        pass
    else:
        raise ValueError("The output directory cannot be inside the training workspace")
    if run_dir.exists():
        raise FileExistsError(f"Output directory already exists: {run_dir}")
    shutil.copytree(workspace, run_dir)

    config_path = _resolve_inside(run_dir, input_config, "input_config")
    if not config_path.is_file():
        raise FileNotFoundError(f"DeePMD input config not found: {config_path}")
    if config_path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValueError("DeePMD input config must be JSON or YAML")
    return run_dir, config_path


def _command_prefix(dp_command: str, backend: DeepMDBackend) -> list[str]:
    prefix = shlex.split(dp_command)
    if not prefix:
        raise ValueError("dp_command cannot be empty")
    return [*prefix, f"--{backend}"]


def _write_log(log_path: Path, command: Sequence[str], result: subprocess.CompletedProcess[str]) -> None:
    command_text = shlex.join(command)
    log_path.write_text(
        f"$ {command_text}\n\n[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}",
        encoding="utf-8",
    )


def _run_cli(command: list[str], run_dir: Path, log_name: str) -> Path:
    log_path = run_dir / log_name
    try:
        result = subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to start DeePMD CLI command {command[0]!r}. "
            "Install DeePMD-kit in the execution environment or set dp_command."
        ) from exc
    _write_log(log_path, command, result)
    if result.returncode != 0:
        raise RuntimeError(
            f"DeePMD CLI exited with status {result.returncode}. See {log_path}"
        )
    return log_path


def _latest_model(run_dir: Path) -> Path | None:
    pretrained_dir = run_dir / "pretrained"
    models = [
        path
        for pattern in ("*.pt", "*.pd", "*.pb")
        for path in run_dir.rglob(pattern)
        if path.is_file() and pretrained_dir not in path.parents
    ]
    return max(models, key=lambda path: path.stat().st_mtime) if models else None


def train_with_cli(
    training_workspace_path: Path,
    input_config: str = "input.json",
    backend: DeepMDBackend = "pt",
    work_path: str = ".",
    output_dir_name: str = "deepmd_train",
    dp_command: str = "dp",
    skip_neighbor_stat: bool = False,
    extra_cli_args: list[str] | None = None,
) -> DeepMDTrainingResult:
    run_dir, config_path = _stage_workspace(
        training_workspace_path, work_path, output_dir_name, input_config
    )
    command = [
        *_command_prefix(dp_command, backend),
        "train",
        str(config_path.relative_to(run_dir)),
    ]
    if skip_neighbor_stat:
        command.append("--skip-neighbor-stat")
    command.extend(extra_cli_args or [])

    log_path = _run_cli(command, run_dir, "deepmd_train.log")
    result: DeepMDTrainingResult = {
        "output_path": run_dir,
        "log_path": log_path,
        "command": shlex.join(command),
    }
    model_path = _latest_model(run_dir)
    if model_path is not None:
        result["model_path"] = model_path
    return result


def _download_pretrained_model(
    run_dir: Path,
    dp_command: str,
    model_name: str,
) -> tuple[Path, Path]:
    cache_dir = run_dir / "pretrained"
    prefix = shlex.split(dp_command)
    if not prefix:
        raise ValueError("dp_command cannot be empty")
    command = [
        *prefix,
        "pretrained",
        "download",
        model_name,
        "--cache-dir",
        str(cache_dir),
    ]
    log_path = _run_cli(command, run_dir, "deepmd_pretrained_download.log")
    models = sorted(cache_dir.glob("*.pt"))
    if len(models) != 1:
        raise RuntimeError(
            f"Expected one downloaded PyTorch model in {cache_dir}, found {len(models)}"
        )
    return models[0], log_path


def finetune_dpa3_with_cli(
    training_workspace_path: Path,
    pretrained_model_path: Path | None = None,
    pretrained_model_name: str = "DPA-3.3-1M",
    input_config: str = "input.json",
    work_path: str = ".",
    output_dir_name: str = "deepmd_dpa3_finetune",
    dp_command: str = "dp",
    model_branch: str | None = None,
    use_pretrain_script: bool = True,
    skip_neighbor_stat: bool = False,
    extra_cli_args: list[str] | None = None,
) -> DeepMDTrainingResult:
    run_dir, config_path = _stage_workspace(
        training_workspace_path, work_path, output_dir_name, input_config
    )

    if pretrained_model_path is not None:
        source_model = Path(pretrained_model_path).resolve()
        if not source_model.is_file():
            raise FileNotFoundError(f"Pretrained DPA-3 model not found: {source_model}")
        if source_model.suffix.lower() != ".pt":
            raise ValueError("DPA-3 fine-tuning requires a PyTorch .pt model")
        staged_model = run_dir / "pretrained" / source_model.name
        staged_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_model, staged_model)
    else:
        if not pretrained_model_name:
            raise ValueError("pretrained_model_name cannot be empty")
        staged_model, _ = _download_pretrained_model(
            run_dir, dp_command, pretrained_model_name
        )

    command = [
        *_command_prefix(dp_command, "pt"),
        "train",
        str(config_path.relative_to(run_dir)),
        "--finetune",
        str(staged_model.relative_to(run_dir)),
    ]
    if use_pretrain_script:
        command.append("--use-pretrain-script")
    if model_branch:
        command.extend(["--model-branch", model_branch])
    if skip_neighbor_stat:
        command.append("--skip-neighbor-stat")
    command.extend(extra_cli_args or [])

    log_path = _run_cli(command, run_dir, "deepmd_finetune.log")
    result: DeepMDTrainingResult = {
        "output_path": run_dir,
        "log_path": log_path,
        "command": shlex.join(command),
        "pretrained_model_path": staged_model,
    }
    model_path = _latest_model(run_dir)
    if model_path is not None and model_path != staged_model:
        result["model_path"] = model_path
    return result
