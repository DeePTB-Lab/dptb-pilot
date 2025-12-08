import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict, Union, List
from pathlib import Path
import subprocess as sp

from dptb.entrypoints.emp_sk import EmpSK
from matplotlib import image as mpimg, pyplot as plt

from dptb_pilot.tools.init import mcp

class ModelResult(TypedDict):
    model_path: str

class BandResult(TypedDict):
    band_path: str

@mcp.tool()
def band_with_sk_model(
    model_file_name: str = "your_model_file_name",
    structure_file_name: str = "your_structure_file_name",
    work_path: str = "."
) -> BandResult:
    """
    使用输入的SK模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_name != "your_model_file_name", "SK模型必须输入"
    assert structure_file_name != "your_structure_file_name", "输入的结构必须输入"
    
    work_dir = Path(work_path).absolute()
    model_path = work_dir / model_file_name
    structure_path = work_dir / structure_file_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy input files to temp dir
        shutil.copy(model_path, temp_path / model_file_name)
        shutil.copy(structure_path, temp_path / structure_file_name)
        
        cmd = ['dptb', 'run', 'band', '-i', model_file_name, '-stu', structure_file_name, '-o', 'band_running']
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"dptb execution failed:\n{result.stderr}")

        img_path = temp_path / 'band_running' / 'results' / 'band.png'
        if not img_path.exists():
             raise RuntimeError("Band calculation finished but no image generated.")

        import time
        timestamp = int(time.time())
        filename = f"band_{timestamp}.png"
        output_img_path = work_dir / filename
        
        plt.figure(figsize=(10, 8))
        img = mpimg.imread(str(img_path))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(output_img_path)
        plt.close()
        
        print(f"Band structure saved to {output_img_path}")

    return {"band_path": str(work_dir), "image_file": filename}
