import json
import os
import shutil
import subprocess
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
from dptb.postprocess.unified import TBSystem

from matplotlib import image as mpimg, pyplot as plt

from dptb_pilot.tools.modules.abacus.core import _abacus_get_fermi_level
from dptb_pilot.tools.modules.util.comm import generate_work_path, temporary_chdir
from dptb_pilot.tools.modules.util.get_dptb_path import get_dptb_path


def parse_kpath_input(input_str):
    """
    将输入字符串转换为kpath_config格式

    参数:
    input_str: 字符串，格式如 "[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"

    返回:
    字典，格式如 {
        "method": "abacus",
        "kpath": [...],
        "klabels": [...]
    }
    """
    import ast

    # 解析输入字符串
    try:
        # 将字符串转换为Python列表
        parsed_list = ast.literal_eval(input_str)
    except:
        # 如果直接解析失败，尝试添加引号
        # 将类似 [0.0,0.0,0.0,50,G] 中的 G 加上引号
        import re
        modified_str = re.sub(r',([A-Za-z]+)\]', r',"\1"]', input_str)
        modified_str = re.sub(r',([A-Za-z]+),', r',"\1",', modified_str)
        parsed_list = ast.literal_eval(modified_str)

    kpath = []
    klabels = []

    for i, item in enumerate(parsed_list):
        # 提取数值部分
        coords = item[:3]  # 前三个是坐标
        num_points = item[3]  # 第四个是点数
        label = item[4]  # 第五个是标签

        # 格式化坐标，保留3位小数
        formatted_coords = [round(coord, 3) for coord in coords]

        # 添加到kpath
        kpath.append(formatted_coords + [num_points])

        # 处理标签
        if i > 0 and label == parsed_list[i - 1][4]:
            # 如果当前标签与前一个相同，添加分隔符
            klabels.append(f"{parsed_list[i - 1][4]}/{label}")
        else:
            klabels.append(label)

    # 构建最终输出
    kpath_config = {
        "method": "abacus",
        "kpath": kpath,
        "klabels": klabels
    }

    return kpath_config

def get_fermi_level(tbsystem, nel_atom, kmesh=None):
    """
    Calculate the Fermi level of a DeePTB ``TBSystem``.

    Parameters
    ----------
    tbsystem : TBSystem
        Initialized DeePTB tight-binding system.
    nel_atom : dict
        Mapping from element symbol to valence electron count.
    kmesh : list, optional
        Monkhorst-Pack mesh used for Fermi-level integration. Defaults to
        ``[5, 5, 5]``.

    Returns
    -------
    float
        Calculated Fermi energy stored on the ``TBSystem``.
    """

    if kmesh is None:
        kmesh = [5, 5, 5]

    tbsystem.set_electrons(nel_atom)
    tbsystem.get_efermi(kmesh=kmesh)

    return tbsystem.efermi

def find_fermi_level(eigenvalues, n_atoms, valence_electrons=4):
    """
    已被遗弃
    根据电子数守恒自动计算费米能级（0K）
    """

    # 展平并排序
    energies = eigenvalues.ravel()
    energies_sorted = np.sort(energies)

    # 总电子数
    N_e = valence_electrons * n_atoms

    # 占据能级数（自旋简并）
    N_occ = int(N_e / 2)

    if N_occ <= 0 or N_occ >= len(energies_sorted):
        raise ValueError("电子数与能级数不匹配")

    # 费米能级取 HOMO 与 LUMO 中点
    E_F = 0.5 * (
        energies_sorted[N_occ - 1] + energies_sorted[N_occ]
    )

    return E_F

from pathlib import Path
import numpy as np

import numpy as np

import numpy as np


def smart_band_gap(eig, fermi_level, atol=1e-3, max_diff=1.0):
    """
    根据能带覆盖区间自动寻找距离费米能级最近的带隙。
    自适应处理不同 k 点能带序号错位/乱序的情况，并具备高级安全警报机制。

    Parameters
    ----------
    eig : array_like
        能带数据，shape=(Nk, Nband)
    fermi_level : float
        费米能级
    atol : float
        判断区间是否连续的容差（单位 eV），即判定为带隙的最小宽度
    max_diff : float
        费米能级周围的安全窗口半径（单位 eV），在此窗口外的带隙将触发风险评估

    Returns
    -------
    dict or None
        包含带隙信息的字典。若不存在任何带隙，则返回 None。
    """
    eigenvalues = np.asarray(eig)
    if atol is None:
        atol = 1e-3

    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues should have shape (Nk, Nband)")

    # 1. 将所有 k 点的本征值打平并进行全局升序排序
    flat_energies = np.sort(eigenvalues.flatten())

    # 记录全局能量边界，用于后续屏蔽边缘带隙
    global_min = flat_energies[0]
    global_max = flat_energies[-1]
    edge_buffer = 0.2

    # 2. 计算相邻本征值之间的能量差
    diffs = np.diff(flat_energies)

    # 3. 找出所有大于容差 atol 的间隙索引
    gap_indices = np.where(diffs > atol)[0]

    # 如果没有任何间隙，使用更细的带隙搜索方式
    if len(gap_indices) == 0:
        return narrow_band_gap(eig, fermi_level)

    all_gaps = []
    biggest_gap = None
    max_gap_width = 0.0

    # 4. 收集所有带隙，并定位距离费米能级最近的带隙
    for idx in gap_indices:
        lower = float(flat_energies[idx])
        if lower > fermi_level + max_diff:
            continue
        upper = float(flat_energies[idx + 1])
        if upper < fermi_level - max_diff:
            continue
        all_gaps.append((lower, upper))

        # # 以带隙中心到费米能级的距离作为“最近”的判定依据
        # gap_center = (lower + upper) / 2.0
        # dist_to_center = abs(gap_center - fermi_level)
        #
        # if dist_to_center < min_dist_to_center:
        #     min_dist_to_center = dist_to_center
        #     closest_gap = (lower, upper)

        # 以带隙大小作为判定依据
        gap_width = upper - lower
        if gap_width > max_gap_width:
            max_gap_width = gap_width
            biggest_gap = (lower, upper)

    # 提取目标带隙的核心参数
    if biggest_gap is None:
        return narrow_band_gap(eig, fermi_level)

    gap_lower, gap_upper = biggest_gap
    gap_width = gap_upper - gap_lower
    gap_center = (gap_lower + gap_upper) / 2.0

    # 计算费米能级到带隙的物理距离以及是否在带隙内部
    if gap_lower <= fermi_level <= gap_upper:
        distance_to_fermi = 0.0
        fermi_inside_gap = True
    else:
        distance_to_fermi = float(min(abs(gap_lower - fermi_level), abs(gap_upper - fermi_level)))
        fermi_inside_gap = False

    # 5. 升级版风险检查与单次最危险警报机制
    risky = False
    dangerous_gaps = []

    for g_lower, g_upper in all_gaps:
        # 条件 A: 判断该带隙是否在 fermi_level ± max_diff eV 的窗口外面
        outside_window = (g_upper < fermi_level - max_diff) or (g_lower > fermi_level + max_diff)
        if not outside_window:
            continue

        # 条件 B: 屏蔽能带最外侧边缘 0.2 eV 内的带隙
        near_edge = (g_lower < global_min + edge_buffer) or (g_upper > global_max - edge_buffer)
        if near_edge:
            continue

        # 条件 C: 只有当该带隙大小大于当前找到的目标带隙时，才认为具有威胁
        g_width = g_upper - g_lower
        if g_width > gap_width:
            # 计算该危险带隙到费米能级的距离
            g_dist = float(min(abs(g_lower - fermi_level), abs(g_upper - fermi_level)))
            dangerous_gaps.append({
                "lower": g_lower,
                "upper": g_upper,
                "width": g_width,
                "distance": g_dist
            })

    # 如果存在通过筛选的危险带隙，则触发警报
    if len(dangerous_gaps) > 0:
        risky = True
        # 按照带隙宽度从大到小排序，筛选出“最危险”的那个
        most_dangerous = max(dangerous_gaps, key=lambda x: x["width"])

        print(f"[⚠️ WARNING] 在设定窗口外检测到更宽的潜在带隙！请人工核对费米能级是否准确。")
        print(f"    - 最险带隙大小: {most_dangerous['width']:.4f} eV")
        print(f"    - 距费米能级距离: {most_dangerous['distance']:.4f} eV")
        print(f"    - 带隙能量区间: [{most_dangerous['lower']:.4f}, {most_dangerous['upper']:.4f}] eV")

    return {
        "gap_lower": float(gap_lower),
        "gap_upper": float(gap_upper),
        "gap_width": float(gap_width),
        "gap_center": float(gap_center),
        "distance_to_fermi": distance_to_fermi,
        "fermi_inside_gap": fermi_inside_gap,
        "all_gaps": all_gaps,
        "risky": risky
    }

def narrow_band_gap(eig, fermi_level):
    """
    自适应能带结构计算函数。
    1. 优先尝试全局排序法寻找标准半导体大带隙。
    2. 若带隙小于容差 atol，自动切换为针对金属管/准金属管的本地邻域排他性查找算法。

    Parameters
    ----------
    eig : array_like
        能带数据，shape=(Nk, Nband)
    fermi_level : float
        费米能级
    atol : float
        判定为标准半导体大带隙的容差（单位 eV）
    title_name : str
        绘制能带图时的标题名称（供用户自行传入）
    """
    eigenvalues = eig
    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues should have shape (Nk, Nband)")

    Nk, Nband = eigenvalues.shape

    # =========================================================================
    # 算法二：进入金属管/微带隙管专用精准定位算法
    # =========================================================================
    print(f"[🔵 METALLIC/SMALL-GAP] 全局带隙低于容差，启动金属管特异性局部排他性搜索...")

    E_min, E_max, g_upper, g_lower, is_valid, final_k, g_width  = None, None, None, None, None, None, None

    # =========================================================================
    # 步骤 0：准备搜索在最后一个 k 点（右边界高对称点）的费米能级位置能带以及其上其下向上向下2个能带
    # =========================================================================
    k_start = Nk - 1
    k_energies = np.sort(eigenvalues[k_start])

    # 找到插入位置
    idx = np.searchsorted(k_energies, fermi_level)

    if idx == 0 or idx == Nband:
        return {
            "type": "insufficient_fermi_bracketing",
            "is_valid": False,
            "gap_lower": None,
            "gap_upper": None,
            "gap_width": 0.0,
            "diagnostic_message": (
                "The reference Fermi level is outside the eigenvalue range at "
                "the final k-point; a local gap cannot be tracked reliably."
            ),
        }

    # Try the pair bracketing Ef first, then adjacent pairs on either side.
    checking_list = [
        (idx - 1, idx),
        (idx - 2, idx - 1),
        (idx, idx + 1),
        (idx - 3, idx - 2),
        (idx + 1, idx + 2),
    ]
    checking_list = [
        pair for pair in checking_list
        if 0 <= pair[0] < Nband and 0 <= pair[1] < Nband
    ]

    for checking in checking_list:
        # =========================================================================
        # 步骤 1：在最后一个 k 点（右边界高对称点）初始化 a 和 b
        # =========================================================================

        # 初始化：a 为 0 以上的最小值（导带底），b 为 0 以下的最大值（价带顶）
        a = k_energies[checking[1]]
        b = k_energies[checking[0]]

        final_k = k_start
        final_a = a
        final_b = b

        # =========================================================================
        # 步骤 2：由右向左（从 Nk-2 到 0）逐格滑动追踪
        # =========================================================================
        for k in range(k_start - 1, -1, -1):
            energies = eigenvalues[k]

            # 查找当前 k 点：在 a, b之间的能量点
            band_mid = (a + b) / 2
            cand_a = energies[energies > band_mid]
            cand_b = energies[energies < band_mid]

            # 如果任一侧找不到能继续延伸的平滑能带，触发“搜索不到”停止条件
            if len(cand_a) == 0 or len(cand_b) == 0:
                print(f"[ℹ️ INFO] 在 k={k} 处因无法继续向下/向上延展而停止追踪。")
                break

            # 取最接近能量中心的位置
            next_a = cand_a[np.argmin(cand_a)]
            next_b = cand_b[np.argmin(-cand_b)]

            # 核心停止条件：如果两条追踪能带相交或发生了能量倒挂（对应 X 形交点）
            if next_a <= next_b or (next_a > a and next_b < b):
                final_k = k + 1
                final_a = a
                final_b = b
                print(f"[🎯 TRACKING] 在 k={final_k} 处成功捕捉到能带线性交叉位置！停止追踪。")
                break

            # 未相交则更新状态，继续向左推进
            a = next_a
            b = next_b
            final_k = k
            final_a = a
            final_b = b

        # 确定最终锁定的目标本地带隙区间
        g_lower = min(final_a, final_b)
        g_upper = max(final_a, final_b)
        g_width = g_upper - g_lower
        g_center = (g_lower + g_upper) / 2.0

        # =========================================================================
        # 步骤 3：在最终位置执行 k±2 动态展宽计算与全谱排他性校验
        # =========================================================================
        neighbors = [final_k - 2, final_k - 1, final_k + 1, final_k + 2]
        valid_neighbors = [nb for nb in neighbors if 0 <= nb < Nk]  # 防止索引越界

        val_unders = []
        val_aboves = []

        for nb in valid_neighbors:
            nb_energies = eigenvalues[nb]
            under = nb_energies[nb_energies <= g_center]
            above = nb_energies[nb_energies > g_center]

            if len(under) > 0: val_unders.append(np.max(under))
            if len(above) > 0: val_aboves.append(np.min(above))

        E_min = np.min(val_unders) if len(val_unders) > 0 else g_lower
        E_max = np.max(val_aboves) if len(val_aboves) > 0 else g_upper

        # 全局排他性检查：确保除 [k-3, k+3] 区域外，全谱没有任何其他能带落在 [E_min, E_max] 内
        outside_ks = [i for i in range(Nk) if i < final_k - 3 or i > final_k + 3]
        is_valid = True

        for ok in outside_ks:
            ok_energies = eigenvalues[ok]
            if np.any((ok_energies > E_min) & (ok_energies < E_max)):
                is_valid = False
                break

        # 根据校验状态输出控制台日志
        if not is_valid:
            print(f"[⚠️ WARNING] 校验未通过！在计算展宽区间 [{E_min:.4f}, {E_max:.4f}] eV 内存在其他 k 点的能带污染。")
            print(f"            仍将强制绘制图片，请在图面上观察黄色区间内被污染的红色散点。")
        else:
            print(f"[🎉 SUCCESS] 目标交叉带隙通过全部验证！区间纯净无污染。")

        print(f"    - 精确锁定的 k 点索引: {final_k}")
        print(f"    - 该处局部带隙大小: {g_width:.4f} eV")
        print(f"    - 动态能量展宽墙: [{E_min:.4f}, {E_max:.4f}] eV")
        if is_valid:
            break

    return {
        "type": "backtracked_metallic" if is_valid else "polluted_metallic",
        "is_valid": is_valid,
        "k_target": final_k,
        "gap_lower": float(g_lower),
        "gap_upper": float(g_upper),
        "gap_width": float(g_width),
        "energy_window_min": float(E_min),
        "energy_window_max": float(E_max)
    }

def _load_band_gap_input(band_structure_file_path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Load supported band files and normalize eigenvalues to ``(Nk, Nband)``."""
    path = Path(band_structure_file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Band-structure file not found: {path}")

    data_raw = np.load(path, allow_pickle=True)
    if isinstance(data_raw, np.lib.npyio.NpzFile):
        try:
            data = {key: data_raw[key] for key in data_raw.files}
        finally:
            data_raw.close()
    elif isinstance(data_raw, np.ndarray) and data_raw.dtype == object:
        if data_raw.size != 1:
            raise ValueError(
                "Object-array .npy input must contain exactly one band-data mapping"
            )
        loaded = data_raw.reshape(()).item()
        if not isinstance(loaded, Mapping):
            raise TypeError("Object-array .npy input must contain a mapping")
        data = dict(loaded)
    elif isinstance(data_raw, np.ndarray):
        data = {"eigenvalues": data_raw}
    else:
        raise TypeError(f"Unsupported band-structure payload: {type(data_raw).__name__}")

    if "eigenvalues" not in data:
        raise KeyError("Band-structure data does not contain 'eigenvalues'")

    try:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("'eigenvalues' must be a numeric array") from exc

    if eigenvalues.ndim == 1:
        eigenvalues = eigenvalues.reshape(1, -1)
    elif eigenvalues.ndim == 3 and eigenvalues.shape[0] == 1:
        eigenvalues = eigenvalues[0]

    if eigenvalues.ndim != 2:
        raise ValueError(
            "'eigenvalues' must have shape (Nk, Nband), (Nband,), or "
            f"(1, Nk, Nband); received {eigenvalues.shape}"
        )
    if 0 in eigenvalues.shape:
        raise ValueError("'eigenvalues' must not be empty")

    if "xlist" in data:
        xlist_size = np.asarray(data["xlist"]).size
        if eigenvalues.shape[0] == xlist_size:
            pass
        elif eigenvalues.shape[1] == xlist_size:
            eigenvalues = eigenvalues.T
        else:
            raise ValueError(
                "'xlist' length does not match either eigenvalue dimension: "
                f"xlist={xlist_size}, eigenvalues={eigenvalues.shape}"
            )

    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("'eigenvalues' contains NaN or infinite values")

    return data, eigenvalues


def _as_finite_scalar(value: Any, name: str) -> float:
    """Convert a Python or NumPy scalar to a finite float."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar; received shape {array.shape}")
    try:
        scalar = float(array.reshape(()).item())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _energy_indices(eigenvalues: np.ndarray, energy: Optional[float]):
    if energy is None:
        return None, None
    flat_index = int(np.argmin(np.abs(eigenvalues - energy)))
    kpoint_index, band_index = np.unravel_index(flat_index, eigenvalues.shape)
    return int(kpoint_index), int(band_index)


def _normalize_gap_candidates(all_gaps) -> list[Dict[str, float]]:
    candidates = []
    for gap in all_gaps or []:
        if isinstance(gap, Mapping):
            lower = float(gap["lower"])
            upper = float(gap["upper"])
        else:
            lower, upper = map(float, gap)
        candidates.append({
            "lower": lower,
            "upper": upper,
            "width": max(0.0, upper - lower),
        })
    return candidates


def _unified_band_gap_result(
    *,
    eigenvalues: np.ndarray,
    fermi_level: float,
    fermi_level_source: str,
    calculation_method: str,
    gap_lower: Optional[float],
    gap_upper: Optional[float],
    all_gaps=None,
    risky: bool = False,
    is_valid: bool = True,
    status: str = "ok",
    diagnostic_message: Optional[str] = None,
    smart_result: Optional[Mapping[str, Any]] = None,
):
    """Build the stable public result shape shared by all calculation paths."""
    lower = None if gap_lower is None else float(gap_lower)
    upper = None if gap_upper is None else float(gap_upper)
    gap = 0.0 if lower is None or upper is None else max(0.0, upper - lower)
    center = None if lower is None or upper is None else (lower + upper) / 2.0
    inside = None if center is None else lower <= fermi_level <= upper
    distance = None
    if center is not None:
        distance = 0.0 if inside else min(abs(lower - fermi_level), abs(upper - fermi_level))

    vbm_kpoint_index, vbm_band_index = _energy_indices(eigenvalues, lower)
    cbm_kpoint_index, cbm_band_index = _energy_indices(eigenvalues, upper)
    smart_result = smart_result or {}

    return {
        "band_gap": float(gap),
        "vbm": lower,
        "cbm": upper,
        "vbm_band_index": vbm_band_index,
        "cbm_band_index": cbm_band_index,
        "vbm_kpoint_index": vbm_kpoint_index,
        "cbm_kpoint_index": cbm_kpoint_index,
        "gap_lower": lower,
        "gap_upper": upper,
        "gap_width": float(gap),
        "gap_center": center,
        "fermi_level": float(fermi_level),
        "fermi_level_source": fermi_level_source,
        "calculation_method": calculation_method,
        "distance_to_fermi": distance,
        "fermi_inside_gap": inside,
        "all_gaps": _normalize_gap_candidates(all_gaps),
        "risky": bool(risky),
        "is_valid": bool(is_valid),
        "status": status,
        "diagnostic_message": diagnostic_message,
        "k_target": smart_result.get("k_target"),
        "energy_window_min": smart_result.get("energy_window_min"),
        "energy_window_max": smart_result.get("energy_window_max"),
    }


def _band_gap(band_structure_file_path: Path,
              fermi_level: Optional[float] = None,
              n_atoms: Optional[int] = None,
              pseudo_fermi_level: Optional[float] = None,
              calculation_method: Literal["auto", "fermi", "smart"] = "auto",
              smart_band_gap_atol: Optional[float] = None,
              smart_band_gap_max_diff: float = 1.0):
    """
    Calculate a band gap from a normalized DeePTB band-structure input.

    Parameters
    ----------
    band_structure_file_path : Path
        Path to ``.npz`` data, a ``.npy`` mapping, or a numeric ``.npy`` array.
        The mapping form must contain ``eigenvalues`` and may contain
        ``fermi_level`` or ``E_fermi``. Pickled object-array files must only come
        from trusted sources.
    fermi_level : float, optional
        Explicit Fermi level used to split occupied and unoccupied states.
    n_atoms : int, optional
        Number of atoms used only when neither an argument nor the file supplies
        a Fermi level.
    pseudo_fermi_level : float, optional
        Approximate reference energy. In ``auto`` mode, providing this value
        selects the smart/local-gap algorithm. Mutually exclusive with
        ``fermi_level``. Zero is a valid value.
    calculation_method : {"auto", "fermi", "smart"}, optional
        ``fermi`` splits all states at the reference energy. ``smart`` searches
        for a gap near it. ``auto`` preserves the legacy convention: use
        ``smart`` only when ``pseudo_fermi_level`` is explicitly provided.
    smart_band_gap_atol : float, optional
        Minimum resolved gap for smart search, in eV. Defaults to ``1e-3``.
    smart_band_gap_max_diff : float, optional
        Search radius around the reference energy, in eV. Defaults to ``1.0``.

    Returns
    -------
    dict
        A stable dictionary for every calculation path. ``band_gap`` and
        ``gap_width`` are identical; ``vbm``/``cbm`` are identical to
        ``gap_lower``/``gap_upper`` and use the input energy reference. Missing
        edges and indices are returned as ``None`` rather than omitted. The
        result also reports the Fermi-level source, selected method, validity,
        status, and smart-search diagnostics.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    TypeError, ValueError, KeyError
        If the payload, eigenvalue shape, scalar parameters, or method is invalid.
    """
    if calculation_method not in {"auto", "fermi", "smart"}:
        raise ValueError("calculation_method must be 'auto', 'fermi', or 'smart'")
    if fermi_level is not None and pseudo_fermi_level is not None:
        raise ValueError("fermi_level and pseudo_fermi_level are mutually exclusive")
    if calculation_method == "fermi" and pseudo_fermi_level is not None:
        raise ValueError("pseudo_fermi_level cannot be used with calculation_method='fermi'")
    if n_atoms is not None and (
        isinstance(n_atoms, bool)
        or not isinstance(n_atoms, (int, np.integer))
        or n_atoms <= 0
    ):
        raise ValueError("n_atoms must be a positive integer")

    data, eig = _load_band_gap_input(band_structure_file_path)
    use_smart_band_gap = (
        calculation_method == "smart"
        or calculation_method == "auto" and pseudo_fermi_level is not None
    )

    if pseudo_fermi_level is not None:
        reference_fermi_level = pseudo_fermi_level
        fermi_level_source = "argument:pseudo_fermi_level"
    elif fermi_level is not None:
        reference_fermi_level = fermi_level
        fermi_level_source = "argument:fermi_level"
    elif data.get("fermi_level") is not None:
        reference_fermi_level = data["fermi_level"]
        fermi_level_source = "file:fermi_level"
    elif data.get("E_fermi") is not None:
        reference_fermi_level = data["E_fermi"]
        fermi_level_source = "file:E_fermi"
    else:
        if n_atoms is None:
            raise ValueError(
                "No Fermi level was provided; pass fermi_level, "
                "pseudo_fermi_level, or n_atoms, or include fermi_level/E_fermi "
                "in the band file"
            )
        reference_fermi_level = find_fermi_level(eig, n_atoms)
        fermi_level_source = "estimated:n_atoms"

    reference_fermi_level = _as_finite_scalar(reference_fermi_level, "fermi_level")

    if not use_smart_band_gap:
        energies = eig.ravel()
        valence_mask = energies <= reference_fermi_level
        conduction_mask = energies > reference_fermi_level
        if not np.any(valence_mask) or not np.any(conduction_mask):
            return _unified_band_gap_result(
                eigenvalues=eig,
                fermi_level=reference_fermi_level,
                fermi_level_source=fermi_level_source,
                calculation_method="fermi",
                gap_lower=None,
                gap_upper=None,
                risky=True,
                is_valid=False,
                status="fermi_level_outside_spectrum",
                diagnostic_message=(
                    "The reference Fermi level does not split the supplied "
                    "eigenvalues into occupied and unoccupied states."
                ),
            )

        vbm = float(np.max(energies[valence_mask]))
        cbm = float(np.min(energies[conduction_mask]))
        return _unified_band_gap_result(
            eigenvalues=eig,
            fermi_level=reference_fermi_level,
            fermi_level_source=fermi_level_source,
            calculation_method="fermi",
            gap_lower=vbm,
            gap_upper=cbm,
        )

    atol = 1e-3 if smart_band_gap_atol is None else _as_finite_scalar(
        smart_band_gap_atol, "smart_band_gap_atol"
    )
    max_diff = _as_finite_scalar(smart_band_gap_max_diff, "smart_band_gap_max_diff")
    if atol < 0 or max_diff < 0:
        raise ValueError("smart-band-gap tolerances must be non-negative")

    smart_result = smart_band_gap(
        eig,
        reference_fermi_level,
        atol=atol,
        max_diff=max_diff,
    )
    if smart_result is None:
        smart_result = {}
    gap_lower = smart_result.get("gap_lower")
    gap_upper = smart_result.get("gap_upper")
    is_valid = bool(smart_result.get("is_valid", gap_lower is not None and gap_upper is not None))
    return _unified_band_gap_result(
        eigenvalues=eig,
        fermi_level=reference_fermi_level,
        fermi_level_source=fermi_level_source,
        calculation_method="smart",
        gap_lower=gap_lower,
        gap_upper=gap_upper,
        all_gaps=smart_result.get("all_gaps"),
        risky=bool(smart_result.get("risky", not is_valid)),
        is_valid=is_valid,
        status="ok" if is_valid else smart_result.get("type", "gap_not_found"),
        diagnostic_message=smart_result.get("diagnostic_message"),
        smart_result=smart_result,
    )


def _band_predict(
        model_file_path: Path,
        structure_file_path: Path,
        kpath: str,
        nel_atom: Dict[str, int],
        kmesh: str = None,
        override_overlap: Path = None,
        smearing_method: str = "FD",
        sigma_val: float = None,
        work_path: str = "."
    ):
    """
    使用输入的模型预测结构能带。

    方法选择建议:
        这是普通（非 Julia）能带计算方法。对于结构较小，或费米能级不够准确的情况，
        优先使用本方法，结果更稳妥。只有在确实需要处理大结构时，才考虑
        ``band_predict_with_julia``；Julia 方法虽然速度更快，但只能得到局域能带。

    参数:
        model_file_path: 使用的 model 路径。
        structure_file_path: 输入的结构文件路径。结构文件应为 VASP 格式。
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        smearing_method: 计算费米能级时使用的展宽方法，默认 ``FD``。
        sigma_val: 计算费米能级时使用的展宽宽度；默认使用 DeePTB 配置。
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_path, "模型必须输入"
    assert structure_file_path, "输入的结构必须输入"

    _work_path = Path(generate_work_path()).absolute()
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not structure_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    import tempfile

    with tempfile.TemporaryDirectory(dir=_work_path):
        tbsystem = TBSystem(data=str(structure_file_path),
                            calculator=str(model_file_path),
                            override_overlap=str(override_overlap) if override_overlap else None)
        kpath_config = parse_kpath_input(kpath)

        tbsystem.set_electrons(nel_atom=nel_atom)
        if kmesh:
            import ast
            kmesh = ast.literal_eval(kmesh)
        else:
            kmesh = [5,5,5]
        fermi_level = tbsystem.get_efermi(kmesh=kmesh,
                                          smearing_method=smearing_method,
                                          sigma_val=sigma_val)

        tbsystem.band.set_kpath(**kpath_config)
        band_data = tbsystem.band.compute()

        import time
        timestamp = int(time.time())
        bandstructure_filename = f"bandstructure_{timestamp}"
        output_bandstructure_path = work_dir / bandstructure_filename
        band_data.export(output_bandstructure_path)
        output_bandstructure_path = output_bandstructure_path.with_suffix(".npz")
        if not output_bandstructure_path.is_file():
            raise RuntimeError(f"Band structure export did not create {output_bandstructure_path}")

        band_img_filename = f"band_{timestamp}.png"
        output_band_img_path = work_dir / band_img_filename
        band_data.plot(filename=output_band_img_path,
                       emin=-25,emax=8)

    return {"band_structure_file_path": output_bandstructure_path,
            "image_file_path": output_band_img_path,
            "fermi_level": fermi_level}

def _band_predict_with_julia(
        model_file_path: Path,
        structure_file_path: Path,
        kpath: str,
        nel_atom: Dict[str, int],
        kmesh: str = None,
        override_overlap: Path = None,
        eig_solver: str = "numpy",
        work_path: str = ".",
        emin: float = None,
        emax: float = None,
        fermi_level: float = None,
        pseudo_fermi_level: float = None,
        julia_script_path: Path = None,
        num_band: int = 30
    ):
    """
    【重要】非默认能带计算方法：仅在下述适用条件全部满足时使用 Julia 稀疏求解器。
    使用本方法预测结构的局域能带。

    重要的适用范围与选择规则:
        Julia 方法只能得到局域能带，但计算速度远快于普通方法。对于结构较小，
        或费米能级不准确的情况，不要使用本方法；此时应使用普通的
        ``_band_predict``，结果更保险。

        本方法仅适用于同时满足以下条件的情况：
        1. 结构较大，普通方法的计算成本过高；
        2. 费米能级十分准确（由外部输入，或模型准确度极高）；
        3. 只需要局域能带来获得带隙等信息，而不需要完整能带结构。

        Agent 不应仅因为 Julia 方法更快就默认选择它；如果结构大小或费米能级
        可靠性不明确，应选择普通的 ``_band_predict``。

    参数:
        model_file_path: 使用的 model 路径。
        structure_file_path: 输入的结构文件路径。结构文件应为 VASP 格式。
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        eig_solver: 为兼容 Python 工作流保留的本征值求解器标签。
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。
        emin: 绘图能量窗口下限。
        emax: 绘图能量窗口上限。
        fermi_level: 精确费米能级；不能与 ``pseudo_fermi_level`` 同时提供。
        pseudo_fermi_level: 近似费米能级；不能与 ``fermi_level`` 同时提供。
        julia_script_path: Julia 稀疏求解脚本路径；默认使用 DeePTB 内置脚本。
        num_band: 稀疏求解器计算的能带数，默认 30。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_path, "模型必须输入"
    assert structure_file_path, "输入的结构必须输入"
    assert not (fermi_level and pseudo_fermi_level), "费米能级与粗费米能级不该同时输入！"

    if pseudo_fermi_level:
        fermi_level = pseudo_fermi_level

    if not emin:
        emin=-1e6
    if not emax:
        emax=1e6

    _work_path = Path(generate_work_path()).absolute()
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not structure_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    structure_file_path = structure_file_path.absolute()
    model_file_path = model_file_path.absolute()
    if julia_script_path:
        julia_script_path = julia_script_path.absolute()

    with tempfile.TemporaryDirectory(dir=_work_path) as temp_dir:
        temp_path = Path(temp_dir)
        tbsystem = TBSystem(data=str(structure_file_path),
                            calculator=str(model_file_path),
                            override_overlap=str(override_overlap) if override_overlap else None)
        kpath_config = parse_kpath_input(kpath)

        tbsystem.set_electrons(nel_atom=nel_atom)
        if kmesh:
            import ast
            kmesh = ast.literal_eval(kmesh)
        else:
            kmesh = [5,5,5]
        """fermi_level = tbsystem.get_efermi(kmesh=kmesh,
                                          eig_solver=eig_solver)"""

        """tbsystem.band.set_kpath(**kpath_config)
        band_data = tbsystem.band.compute(eig_solver=eig_solver)"""

        tbsystem.to_pardiso(output_dir=temp_dir)

        config = {
            "task_options": {
                "task": "band",
                "eig_solver": "numpy",
                "kline_type": "abacus",
                "kpath": kpath_config["kpath"],
                "klabels": kpath_config["klabels"],
                "nel_atom": nel_atom,
                "E_fermi": fermi_level if fermi_level else 0,
                "emin": emin,
                "emax": emax
            },
            "device": "cpu",
            "out_wfc": "false",
            "which_k": 0,
            "max_iter": 400,
            "num_band": num_band,
            "gamma_only": "false",
            "isspinful": "false"
        }

        with open(os.path.join(temp_dir, "band.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Path to the configuration file
        config_path = os.path.join(temp_dir, "band.json")

        # Define where Julia should save the results
        julia_out_dir = os.path.join(temp_dir, "julia_results")
        if not os.path.exists(julia_out_dir):
            os.makedirs(julia_out_dir)

        if julia_script_path:
            julia_script = str(julia_script_path)
        else:
            julia_script = os.path.join(get_dptb_path(), "postprocess/julia/sparse_calc_npy_print.jl")

        # Construct the command
        cmd = [
            "julia",
            julia_script,
            "--input_dir", temp_dir,
            "--output_dir", julia_out_dir,
            "--config", config_path
        ]

        print(f"Running Julia command: {' '.join(cmd)}")
        print("This may take a moment...")

        try:
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Julia Execution Successful!")
        except subprocess.CalledProcessError as e:
            print("Julia script failed with error:")
            print(e.stdout)
            print(e.stderr)

        expected_npy = os.path.join(julia_out_dir, "bandstructure.npy")
        data = np.load(expected_npy, allow_pickle=True).item()
        print("Successfully loaded bandstructure.npy")
        print(f"Data keys: {list(data.keys())}")
        if 'eigenvalues' in data:
            evals = data['eigenvalues']
            print(f"Eigenvalues shape: {evals.shape}")

        # get fermi level
        if fermi_level is None:
            print("fermi level not inputted! calculating fermi level...")
            from dptb.postprocess.unified.utils import calculate_fermi_level
            ef = calculate_fermi_level(eigenvalues=data["eigenvalues"],
                                       total_electrons=tbsystem.total_electrons,
                                       spindeg=1 if hasattr(tbsystem.model, 'soc_param') else 2)
            print(f"calculated fermi level = {ef}")


        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))

        x = data["xlist"]
        evals = data["eigenvalues"] - data["E_fermi"]
        for i in range(evals.shape[1]):
            plt.scatter(x, evals[:, i], s=1, c='r')
        plt.xticks(data["high_sym_kpoints"], data["labels"])
        plt.xlim(0, data["xlist"][-1])
        if emin < -9.9e5:
            emin = evals.min()
        if emax > 9.9e5:
            emax = evals.max()
        plt.ylim(emin, emax)
        plt.ylabel('E - EF (ev)')

        import time
        timestamp = int(time.time())
        band_img_filename = f"band_{timestamp}.png"
        output_band_img_path = work_dir / band_img_filename
        plt.savefig(output_band_img_path, dpi=300)
        plt.close()

        bandstructure_filename = f"bandstructure_{timestamp}.npy"
        output_bandstructure_path = work_dir / bandstructure_filename
        shutil.copy(expected_npy, output_bandstructure_path)

    return {"band_structure_file_path": output_bandstructure_path,
            "image_file_path": output_band_img_path,
            "fermi_level": data["E_fermi"]}

def _band_compare(
        dptb_result_path: Path,
        dft_result_path: Path,
        e_min: float = None,
        e_max: float = None,
        match_fermi_level: bool = False,
        delta: float = None,
):
    """
    DFT 与 DeePTB 能带对比绘图

    Parameters
    ----------
    dptb_result_path : Path
        DeePTB 输出 npz 文件路径
    dft_result_path : Path
        ABACUS 输出目录（包含 BANDS_1.dat）
    e_min : float
        最小能量范围（相对费米能级）
    e_max : float
        最大能量范围（相对费米能级）
    match_fermi_level : bool, optional
        是否分别减去 DFT 和 DeePTB 的费米能级后再进行对比。
    delta : float, optional
        对齐费米能级后施加到 DeePTB 能带上的额外能量平移。

    Returns
    -------
    dict
        {
            "band_compare_path": str(path)
        }
    """

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    work_path = Path(generate_work_path()).absolute()
    work_path.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # 1. 读取 DFT 能带（ABACUS）
    # ==========================================================
    fermi_level_dict = _abacus_get_fermi_level(
        abacus_out_path=dft_result_path
    )
    dft_fermi_level = fermi_level_dict["fermi_level"]

    band_file = dft_result_path / "BANDS_1.dat"
    if not band_file.exists():
        raise FileNotFoundError(
            f"未找到 DFT 能带文件: {band_file}"
        )

    dft_data = []
    with open(band_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split()
            values = list(map(float, parts))
            dft_data.append(values)

    dft_data = np.array(dft_data)

    dft_kpoints = dft_data[:, 1]
    dft_band = dft_data[:, 2:]

    # 相对费米能级
    if match_fermi_level:
        dft_band = dft_band - dft_fermi_level

    # ==========================================================
    # 2. 读取 DPTB 能带
    # ==========================================================
    dptb_result = np.load(dptb_result_path)

    if "eigenvalues" not in dptb_result:
        raise KeyError(
            "DPTB 结果中不存在 'eigenvalues'"
        )

    dptb_band = dptb_result["eigenvalues"]

    if "fermi_level" in dptb_result:
        dptb_fermi_level = float(
            np.squeeze(dptb_result["fermi_level"])
        )
    else:
        dptb_fermi_level = 0.0

    if match_fermi_level:
        dptb_band = dptb_band - dptb_fermi_level
        if delta:
            dptb_band = dptb_band + delta

    # ==========================================================
    # 3. 处理维度
    # ==========================================================
    # 常见格式:
    # (nk, nbands)
    # 或 (1, nk, nbands)

    if dptb_band.ndim == 3:
        # 默认取第一个 spin/channel
        dptb_band = dptb_band[0]

    if dptb_band.ndim != 2:
        raise ValueError(
            f"DPTB eigenvalues 维度异常: {dptb_band.shape}"
        )

    nk_dft = len(dft_kpoints)
    nk_dptb = dptb_band.shape[0]

    # ==========================================================
    # 4. k-path 对齐
    # ==========================================================
    if nk_dft != nk_dptb:
        print(
            f"[Warning] DFT k点数({nk_dft}) "
            f"!= DPTB k点数({nk_dptb})，"
            f"将重新线性映射 DPTB k-path"
        )

        dptb_kpoints = np.linspace(
            dft_kpoints.min(),
            dft_kpoints.max(),
            nk_dptb
        )
    else:
        dptb_kpoints = dft_kpoints.copy()

    # ==========================================================
    # 5. 自动能量范围
    # ==========================================================
    all_band = np.concatenate([
        dft_band.reshape(-1),
        dptb_band.reshape(-1)
    ])

    if e_min is None:
        e_min = float(np.min(all_band))

    if e_max is None:
        e_max = float(np.max(all_band))

    # ==========================================================
    # 6. 绘图
    # ==========================================================
    plt.figure(figsize=(6, 5))

    # -------- DFT --------
    for i in range(dft_band.shape[1]):
        plt.plot(
            dft_kpoints,
            dft_band[:, i],
            color="black",
            linewidth=1,
            label="DFT" if i == 0 else None
        )

    # -------- DeePTB --------
    for i in range(dptb_band.shape[1]):
        plt.plot(
            dptb_kpoints,
            dptb_band[:, i],
            color="red",
            linewidth=1,
            linestyle="--",
            alpha=0.85,
            label="DeePTB" if i == 0 else None
        )

    # 费米能级
    if match_fermi_level:
        plt.axhline(
            0,
            linestyle="--",
            linewidth=1
        )
        if delta:
            plt.axhline(
                delta,
                linestyle="--",
                linewidth=1,
                label="DeePTB Shift"
            )
    else:
        plt.axhline(
            dft_fermi_level,
            linestyle="--",
            linewidth=1,
            color="black",
            label="DFT $E_F$"
        )
        plt.axhline(
            dptb_fermi_level,
            linestyle="--",
            linewidth=1,
            color="red",
            label="DeePTB $E_F$"
        )

    plt.ylim(e_min, e_max)

    plt.xlabel("k-path")
    ylabel = "$E - E_f$ (eV)" if match_fermi_level else "Energy (eV)"
    plt.ylabel(ylabel)
    plt.title("DFT vs DeePTB Band Structure")

    plt.legend()

    # ==========================================================
    # 7. 保存
    # ==========================================================
    timestamp = int(time.time())

    save_path = (
        work_path /
        f"band_compare_{timestamp}.png"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "band_compare_path": str(save_path)
    }

