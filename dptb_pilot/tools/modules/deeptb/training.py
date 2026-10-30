from pathlib import Path
from typing import Optional

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import DeePTBTrainingResult
from dptb_pilot.tools.modules.deeptb.submodules.training import train_with_cli


@mcp.tool()
def train_deeptb_model(
    input_config_path: Path,
    training_dataset_path: Path,
    validation_dataset_path: Optional[Path] = None,
    reference_dataset_path: Optional[Path] = None,
    work_path: str = ".",
    output_dir_name: str = "deeptb_train",
    dptb_command: str = "dptb",
    init_model_path: Optional[Path] = None,
    restart_model_path: Optional[Path] = None,
    extra_cli_args: Optional[list[str]] = None,
) -> DeePTBTrainingResult:
    """Submit a DeePTB SK or E3 training run with its dftio dataset.

    The configuration and every supplied dataset/checkpoint ``Path`` are forwarded
    with the remote job. On the execution host, this tool automatically discovers
    their new resolved paths, copies the datasets into a self-contained run
    directory, and rewrites ``data_options.*.root`` before invoking ``dptb train``.
    SK and E3 use the same CLI workflow, so one tool supports both; the requested
    data are inferred from each section's ``get_Hamiltonian``, ``get_overlap``,
    ``get_DM``, and ``get_eigenvalues`` flags.

    Parameters
    ----------
    input_config_path : Path
        DeePTB JSON training input. Its original dataset roots may be local paths;
        they are replaced automatically on the execution host.
    training_dataset_path : Path
        Root of a dftio-processed training dataset. The directory is transferred
        with the job and must contain folders matching ``data_options.train.prefix``.
    validation_dataset_path : Path, optional
        Root used for ``data_options.validation``. If omitted, the transferred
        training dataset root is reused and checked with the validation prefix.
    reference_dataset_path : Path, optional
        Root used for ``data_options.reference``. If omitted, the transferred
        training dataset root is reused and checked with the reference prefix.
    work_path : str, optional
        Parent directory for the generated remote or local training run.
    output_dir_name : str, optional
        Name of the self-contained run directory created under ``work_path``.
    dptb_command : str, optional
        DeePTB executable command, including an optional launcher or environment
        wrapper.
    init_model_path : Path, optional
        Checkpoint used with ``dptb train --init-model``. It is transferred and
        staged inside the run directory.
    restart_model_path : Path, optional
        Checkpoint used with ``dptb train --restart``. It cannot be combined with
        ``init_model_path``.
    extra_cli_args : list[str], optional
        Additional arguments appended to the ``dptb train`` command.

    Returns
    -------
    DeePTBTrainingResult
        Paths to the rewritten config, training output, log, and latest ``.pth``
        checkpoint when one is produced. The dataset-bearing run directory is not
        returned as an artifact, avoiding an unnecessary download of the input data.

    Raises
    ------
    ValueError
        If the config is invalid or a dftio dataset lacks a data type enabled by
        its corresponding ``data_options`` section.
    FileNotFoundError
        If the config or an optional checkpoint does not exist.
    RuntimeError
        If the DeePTB CLI cannot start or exits unsuccessfully.

    Notes
    -----
    Use the automatically exposed ``submit_train_deeptb_model`` entry point for
    asynchronous remote execution. Dataset validation happens before training, so
    an E3 input requesting Hamiltonian/overlap data cannot accidentally train on an
    eigenvalue-only SK dataset, and vice versa.
    """
    return train_with_cli(
        input_config_path=input_config_path,
        training_dataset_path=training_dataset_path,
        validation_dataset_path=validation_dataset_path,
        reference_dataset_path=reference_dataset_path,
        work_path=work_path,
        output_dir_name=output_dir_name,
        dptb_command=dptb_command,
        init_model_path=init_model_path,
        restart_model_path=restart_model_path,
        extra_cli_args=extra_cli_args,
    )
