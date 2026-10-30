import ast
import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict, Union, List
from pathlib import Path
import subprocess as sp

from dptb.entrypoints.emp_sk import EmpSK
from matplotlib import image as mpimg, pyplot as plt

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import RunNegfResult


@mcp.tool()
def run_negf(
        model_file_path: Path,
        config_file_path: Path,
        work_path: str = "."
) -> RunNegfResult:
    """
    使用基准模型预测结构能带。

    参数:
        model_file_path: 使用的DeePTB模型。
        config_file_path: 输入的NEGF配置文件
        work_path: 工作路径。注意应该是文件夹而不是文件。

    返回:
        NEGF结果文件。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_path, "模型必须输入"
    assert config_file_path, "negf的配置文件必须输入"

    # Resolve absolute paths
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {config_file_path}")

    import tempfile
    import shutil

    # Use a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy structure file to temp dir
        shutil.copy(config_file_path, temp_path / config_file_path.name)

        # Run dptb command in temp dir
        cmd = ['dptb', 'run', 'band', '-i', model_file_path.name, '-stu', config_file_path.name, '-o', 'band_running']
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"dptb execution failed:\n{result.stderr}")

        # Check if result image exists
        img_path = temp_path / 'band_running' / 'results' / 'band.png'
        if not img_path.exists():
            raise RuntimeError("Band calculation finished but no image generated.")

        # Copy result back to work_path
        import time
        timestamp = int(time.time())
        filename = f"band_{timestamp}.png"
        output_img_path = work_dir / filename

        plt.figure(figsize=(10, 8))
        img = mpimg.imread(str(img_path))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()  # Close figure to free memory

        print(f"Band structure saved to {output_img_path}")

    return {"band_path": str(work_dir), "image_file": filename}

