from pathlib import Path
from typing import Literal, Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deepmd.cli import finetune_dpa3_with_cli, train_with_cli
from dptb_pilot.tools.modules.deepmd.results import DeepMDTrainingResult


@mcp.tool()
def train_deepmd_model(
    training_workspace_path: Path,
    input_config: str = "input.json",
    backend: Literal["pt", "jax", "pd", "tf"] = "pt",
    work_path: str = ".",
    output_dir_name: str = "deepmd_train",
    dp_command: str = "dp",
    skip_neighbor_stat: bool = False,
    extra_cli_args: Optional[list[str]] = None,
) -> DeepMDTrainingResult:
    """Train a DeePMD model by submitting the native ``dp train`` CLI workflow.

    ``training_workspace_path`` must be a directory containing the DeePMD JSON/YAML
    config and every dataset referenced by its relative paths. The complete directory
    is staged into an isolated output directory, which keeps the same layout for local,
    Slurm, and Bohrium execution. Use the automatically exposed
    ``submit_train_deepmd_model`` tool for asynchronous execution.

    Parameters
    ----------
    training_workspace_path : Path
        Directory containing the input config and all referenced datasets.
    input_config : str, optional
        Config path relative to ``training_workspace_path``.
    backend : {"pt", "jax", "pd", "tf"}, optional
        DeePMD backend selected through the matching CLI flag.
    work_path : str, optional
        Parent directory for the staged training run.
    output_dir_name : str, optional
        Name of the new run directory under ``work_path``.
    dp_command : str, optional
        DeePMD executable command, including an optional launcher.
    skip_neighbor_stat : bool, optional
        Pass ``--skip-neighbor-stat`` to ``dp train``.
    extra_cli_args : list[str], optional
        Additional arguments appended to the training command.

    Returns
    -------
    DeepMDTrainingResult
        Output directory, log, executed command, and model path when produced.
    """
    return train_with_cli(
        training_workspace_path=training_workspace_path,
        input_config=input_config,
        backend=backend,
        work_path=work_path,
        output_dir_name=output_dir_name,
        dp_command=dp_command,
        skip_neighbor_stat=skip_neighbor_stat,
        extra_cli_args=extra_cli_args,
    )


@mcp.tool()
def finetune_dpa3_model(
    training_workspace_path: Path,
    pretrained_model_path: Optional[Path] = None,
    pretrained_model_name: str = "DPA-3.3-1M",
    input_config: str = "input.json",
    work_path: str = ".",
    output_dir_name: str = "deepmd_dpa3_finetune",
    dp_command: str = "dp",
    model_branch: Optional[str] = None,
    use_pretrain_script: bool = True,
    skip_neighbor_stat: bool = False,
    extra_cli_args: Optional[list[str]] = None,
) -> DeepMDTrainingResult:
    """Fine-tune a DeePMD DPA-3 foundation model through the native PyTorch CLI.

    Provide ``pretrained_model_path`` for a local ``.pt`` checkpoint, or omit it to
    download ``pretrained_model_name`` with ``dp pretrained download``. The default is
    the latest built-in DPA-3 model in DeePMD-kit at implementation time,
    ``DPA-3.3-1M``. Fine-tuning runs as ``dp --pt train ... --finetune ...`` and uses
    the pretrained model script by default. Use ``submit_finetune_dpa3_model`` for
    asynchronous remote execution.

    Parameters
    ----------
    training_workspace_path : Path
        Directory containing the input config and all referenced datasets.
    pretrained_model_path : Path, optional
        Local PyTorch ``.pt`` checkpoint. When omitted, the named model is downloaded.
    pretrained_model_name : str, optional
        Built-in DeePMD pretrained model name used when no local checkpoint is given.
    input_config : str, optional
        Config path relative to ``training_workspace_path``.
    work_path : str, optional
        Parent directory for the staged fine-tuning run.
    output_dir_name : str, optional
        Name of the new run directory under ``work_path``.
    dp_command : str, optional
        DeePMD executable command, including an optional launcher.
    model_branch : str, optional
        Model branch passed through ``--model-branch``.
    use_pretrain_script : bool, optional
        Pass ``--use-pretrain-script`` to reuse the pretrained model script.
    skip_neighbor_stat : bool, optional
        Pass ``--skip-neighbor-stat`` to ``dp train``.
    extra_cli_args : list[str], optional
        Additional arguments appended to the fine-tuning command.

    Returns
    -------
    DeepMDTrainingResult
        Output directory, logs, executed command, pretrained checkpoint, and
        fine-tuned model path when produced.
    """
    return finetune_dpa3_with_cli(
        training_workspace_path=training_workspace_path,
        pretrained_model_path=pretrained_model_path,
        pretrained_model_name=pretrained_model_name,
        input_config=input_config,
        work_path=work_path,
        output_dir_name=output_dir_name,
        dp_command=dp_command,
        model_branch=model_branch,
        use_pretrain_script=use_pretrain_script,
        skip_neighbor_stat=skip_neighbor_stat,
        extra_cli_args=extra_cli_args,
    )
