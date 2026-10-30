from pathlib import Path

import numpy as np
from dptb.postprocess.unified import TBSystem


def _hamiltonian_predict(
        model_file_path: Path,
        structure_file_path: Path,
        k_points: str,
        override_overlap: Path = None,
        work_path: str = "."
    ):
    """
    使用输入的模型预测结构哈密顿量。

    参数:
        model_file_path: 使用的 model 路径。
        structure_file_path: 输入的结构文件路径。结构文件应为 VASP 格式。
        k_points: 想要计算的k点，格式如"[[0,0,0],[0,0,0.5]]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        work_path: 哈密顿量信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_path, "模型必须输入"
    assert structure_file_path, "输入的结构必须输入"

    work_dir = Path(work_path).absolute()

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not structure_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    import tempfile

    with tempfile.TemporaryDirectory(dir=work_path):
        tbsystem = TBSystem(data=str(structure_file_path),
                            calculator=str(model_file_path),
                            override_overlap=str(override_overlap) if override_overlap else None)
        import ast
        k_points = ast.literal_eval(k_points)

        hk, sk = tbsystem.calculator.get_hk(atomic_data=tbsystem.data,
                                            k_points=k_points)

        import time
        timestamp = int(time.time())
        hamiltonian_filename = f"predicted_hamiltonian_{timestamp}"
        output_hamiltonian_path = work_dir / hamiltonian_filename
        overlap_filename = f"predicted_overlap_{timestamp}"
        output_overlap_path = work_dir / overlap_filename

        np.save(output_hamiltonian_path, hk.numpy())
        np.save(output_overlap_path, sk.numpy())

    return {
        "hamiltonian_file_path": output_hamiltonian_path.with_suffix(".npy"),
        "overlap_file_path": output_overlap_path.with_suffix(".npy"),
    }
