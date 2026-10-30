import ast
import logging
import os
from pathlib import Path
from typing import TypedDict, List

import dptb.nn.dftb as dftb
from dptb.entrypoints.emp_sk import EmpSK
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
from dptb.utils.argcheck import normalize_run
from dptb.utils.auto_band_config import auto_band_config

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import BandResult, ModelResult
from dptb_pilot.tools.modules.util.comm import generate_work_path

log = logging.getLogger(__name__)

def _band_with_baseline_model(
    structure_file_path: Path,
    basemodel: str = "poly4",
    fermi_level: float = None,
    emin: float = None,
    emax: float = None,
    get_fermi: bool = False,
    kmesh: List[int] = None
) -> BandResult:
    """
    使用基准模型预测结构能带。

    参数:
        structure_file_path: 输入的结构文件路径（Path，应通过get_file_path得到）。结构文件应为vasp的格式
        basemodel: 使用的baseline model类型。可选：poly4，poly2
        fermi_level: 指定使用的费米能级
        emin: 能带下限
        emax: 能带上限
        get_fermi: 是否自动获取费米能级
        kmesh: 如果需要自动获取费米能级，使用的k点网格

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    if kmesh is None:
        kmesh = [5, 5, 5]
    assert structure_file_path, "输入的结构必须输入"

    work_path = Path(generate_work_path()).absolute()

    import tempfile

    # Use a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        jdata, com_opts = auto_band_config(structure=str(structure_file_path), kpathtype='vasp')
        jdata = normalize_run(jdata)
        assert basemodel in ['poly2', 'poly4'], f'init_model {basemodel} is not supported.'
        modelname = f'base_{basemodel}.pth'
        basemodel = os.path.join(os.path.dirname(dftb.__file__), modelname)

        in_common_options = {}
        in_common_options.update(com_opts)
        if jdata.get("device", None):
            in_common_options.update({"device": jdata["device"]})
        if jdata.get("dtype", None):
            in_common_options.update({"dtype": jdata["dtype"]})

        model = build_model(checkpoint=basemodel, common_options=in_common_options)

        import time
        timestamp = int(time.time())
        results_path = work_path / f"band_with_baseline_model_{timestamp}"

        bcal = Band(model=model, results_path=str(results_path), use_gui=False, device=model.device)
        bcal.get_bands(data=str(structure_file_path),
                       kpath_kwargs=jdata["task_options"],
                       pbc=jdata["pbc"],
                       AtomicData_options=jdata['AtomicData_options'])

        bcal.band_plot(ref_band=jdata["task_options"].get("ref_band", None),
                       E_fermi=jdata["task_options"].get("E_fermi", None),
                       emin=emin,
                       emax=emax)

        # Check if result image exists
        bandstructure_path = results_path / 'results' / 'bandstructure.npy'
        if not bandstructure_path.exists():
             raise RuntimeError("Band calculation failed.")
        img_path = results_path / 'results' / 'band.png'
        if not img_path.exists():
             raise RuntimeError("Band calculation finished but no image generated.")

    return {"band_structure_file_path": Path(bandstructure_path),
            "image_file_path": Path(img_path),
            "fermi_level": 0}

def _generate_sk_baseline_model(
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

    return {"model_path": Path(work_path).absolute()}
