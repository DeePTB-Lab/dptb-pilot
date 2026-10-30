import logging
from pathlib import Path
from typing import List

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import BandResult, ModelResult
from dptb_pilot.tools.modules.deeptb.submodules.sk_baseline_model import _band_with_baseline_model, \
    _generate_sk_baseline_model

log = logging.getLogger(__name__)


@mcp.tool()
def band_with_baseline_model(
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

    return _band_with_baseline_model(structure_file_path=structure_file_path,
                                     basemodel=basemodel,
                                     fermi_level=fermi_level,
                                     emin=emin,
                                     emax=emax,
                                     get_fermi=get_fermi,
                                     kmesh=kmesh)


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

    return _generate_sk_baseline_model(basemodel=basemodel,
                                       basis=basis,
                                       work_path=work_path)
