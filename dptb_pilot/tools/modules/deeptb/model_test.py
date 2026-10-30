from pathlib import Path

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import HamiltonianTestResult
from dptb_pilot.tools.modules.deeptb.submodules.model_test import _hamiltonian_test


@mcp.tool()
def hamiltonian_test(
        model_path: Path,
        test_dataset_root_path: Path,
        test_dataset_prefix: str,
        get_overlap: bool = True,
        device: str = "cpu",
        onsite_shift: bool = False,
        clean: bool = True
) -> HamiltonianTestResult:
    """
    Evaluate DeePTB Hamiltonian prediction errors on a test dataset.

    Parameters
    ----------
    model_path : Path
        Path to a DeePTB model checkpoint or model JSON file.
    test_dataset_root_path : Path
        Root directory of the test dataset.
    test_dataset_prefix : str
        Prefix used to select test dataset entries.
    get_overlap : bool, optional
        Whether overlap matrices are present and should be evaluated.
    device : str, optional
        Evaluation device, such as ``cpu`` or ``cuda``.
    onsite_shift : bool, optional
        Enable onsite-shift analysis.
    clean : bool, optional
        Remove processed dataset cache directories after evaluation.

    Returns
    -------
    HamiltonianTestResult
        Dictionary with ``stats`` containing MAE/RMSE analysis results.
    """
    return {"stats": _hamiltonian_test(
        model_path=model_path,
        test_dataset_root_path=test_dataset_root_path,
        test_dataset_prefix=test_dataset_prefix,
        get_overlap=get_overlap,
        device=device,
        onsite_shift=onsite_shift,
        clean=clean,
    )}
