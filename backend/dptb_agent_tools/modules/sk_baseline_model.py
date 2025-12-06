import ast
import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict, Union, List
from pathlib import Path
import subprocess as sp

from dptb.entrypoints.emp_sk import EmpSK
from matplotlib import image as mpimg, pyplot as plt

from dptb_agent_tools.init_mcp import mcp

class ModelResult(TypedDict):
    model_path: str

class BandResult(TypedDict):
    band_path: str

@mcp.tool()
def band_with_baseline_model(
    basemodel: str = "poly4",
    structure_file_name: str = "your_structure_file_name",
    work_path: str = "."
) -> BandResult:
    """
    使用基准模型预测结构能带。

    参数:
        basemodel: 使用的baseline model类型。可选：poly4，poly2
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert structure_file_name != "your_structure_file_name", "输入的结构必须输入"
    
    # Resolve absolute paths
    work_dir = Path(work_path).absolute()
    structure_path = work_dir / structure_file_name
    
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_path}")

    import tempfile
    import shutil
    
    # Use a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy structure file to temp dir
        shutil.copy(structure_path, temp_path / structure_file_name)
        
        # Run dptb command in temp dir
        cmd = ['dptb', 'run', 'band', '-i', basemodel, '-stu', structure_file_name, '-o', 'band_running']
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"dptb execution failed:\n{result.stderr}")
            
        # Check if result image exists
        img_path = temp_path / 'band_running' / 'results' / 'band.png'
        if not img_path.exists():
             raise RuntimeError("Band calculation finished but no image generated.")
             
        # Copy result back to work_path
        plt.figure(figsize=(10, 8))
        img = mpimg.imread(str(img_path))
        plt.imshow(img)
        plt.axis('off')
        output_img_path = work_dir / "band.png"
        plt.savefig(output_img_path)
        plt.close() # Close figure to free memory

    return {"band_path": str(work_dir)}

@mcp.tool()
def generate_sk_baseline_model(
        basemodel: str = "poly4",
        basis: str = '{"B": ["s","p","d"],"N": ["s","p","d"]}',
        work_path: str = "."
) -> ModelResult:
    """
    基于基准模型生成 DeePTB-SK 模型。

    参数:
        basemodel: 使用的baseline model类型。可选：poly4，poly2
        basis: 原子基组。格式按python的dict格式，例如{"B": ["s","p","d"],"N": ["s","p","d"]}。此项应该手动输入！
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含模型文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    common_options = {
        "basis": ast.literal_eval(basis)
    }

    EmpSK(common_options, basemodel=basemodel).to_json(outdir=work_path)

    return {"model_path": str(Path(work_path).absolute())}
