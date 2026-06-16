import json
import os
import shutil
import subprocess
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
from dptb.postprocess.unified import TBSystem

from matplotlib import image as mpimg, pyplot as plt

from dptb_pilot.tools.modules.deeptb.submodules.abacus import _abacus_get_efermi
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

def smart_band_gap(eig, pseudo_fermi_level):
    """
    Estimate a band gap by selecting valence/conduction bands around a pseudo Fermi level.

    Parameters
    ----------
    eig : numpy.ndarray
        Eigenvalue array with shape ``(nk, nbands)``.
    pseudo_fermi_level : float
        Approximate Fermi level used to identify the valence band by band-center
        ordering.

    Returns
    -------
    dict
        Band-gap information including ``band_gap`` and, when available, VBM/CBM
        values and band indices.
    """

    nk, nb = eig.shape

    # 每条带平均能量
    band_center = eig.mean(axis=0)

    # 找到 pseudo Ef 下方最近 band
    vb_idx = np.where(band_center < pseudo_fermi_level)[0]

    if len(vb_idx) == 0:
        return {"band_gap": 0.0}

    vb_idx = vb_idx[-1]

    if vb_idx + 1 >= nb:
        return {"band_gap": 0.0}

    val_band = eig[:, vb_idx]
    cond_band = eig[:, vb_idx + 1]

    vbm = val_band.max()
    cbm = cond_band.min()

    gap = cbm - vbm

    if gap < 0:
        gap = 0.0

    return {
        "band_gap": float(gap),
        "vbm": float(vbm),
        "cbm": float(cbm),
        "vbm_band_index": int(vb_idx),
        "cbm_band_index": int(vb_idx + 1),
    }


def _band_gap(band_structure_file_path: Path,
              fermi_level = None,
              n_atoms = None,
              pseudo_fermi_level: float = None):
    """
    Calculate the band gap from a DeePTB band-structure ``.npz`` or ``.npy`` file.

    Parameters
    ----------
    band_structure_file_path : Path
        Path to a saved band-structure file containing an ``eigenvalues`` array.
    fermi_level : float, optional
        Explicit Fermi level used to split occupied and unoccupied states.
    n_atoms : int, optional
        Number of atoms used by the fallback electron-count Fermi-level estimate.
    pseudo_fermi_level : float, optional
        Approximate Fermi level for the band-center based gap estimator. Mutually
        exclusive with ``fermi_level``.

    Returns
    -------
    dict
        Band-gap dictionary. With pseudo Fermi input it may also include VBM/CBM
        values and band indices.
    """

    assert not (fermi_level and pseudo_fermi_level), "费米能级与粗费米能级不该同时输入！"

    data_raw = np.load(band_structure_file_path, allow_pickle=True)

    # ---------- 1. 统一数据结构 ----------
    if isinstance(data_raw, np.lib.npyio.NpzFile):
        # npz → 类 dict
        data = {k: data_raw[k] for k in data_raw.files}
    elif isinstance(data_raw, np.ndarray) and data_raw.dtype == object:
        # npy → array(dict, dtype=object)
        data = data_raw.item()
    elif isinstance(data_raw, dict):
        data = data_raw
    else:
        raise TypeError(f"不支持的数据格式: {type(data_raw)}")

    # ---------- 2. 读取 eigenvalues ----------
    if "eigenvalues" not in data:
        raise KeyError("未找到 eigenvalues")
    eig = data["eigenvalues"]

    # ---------- 3. 处理费米能级 ----------
    if fermi_level is None:
        fermi_level = pseudo_fermi_level

    if fermi_level is None:
        # npz: fermi_level
        # npy: E_fermi
        fermi_level = data.get("fermi_level", None)

    if fermi_level is None:
        fermi_level = data.get("E_fermi", None)

    # 防止 np.array(0) 这种情况
    if isinstance(fermi_level, np.ndarray):
        fermi_level = float(fermi_level)

    if fermi_level is None:
        if n_atoms is None:
            raise ValueError("必须提供 n_atoms 才能自动计算费米能级")
        print("自动查找费米能级——")
        fermi_level = find_fermi_level(eig, n_atoms)
        print(f"费米能级为：{fermi_level}")

    # ---------- 4. 展平能带 ----------
    if not pseudo_fermi_level:
        energies = eig.ravel()

        valence = energies[energies <= fermi_level]
        conduction = energies[energies > fermi_level]

        if valence.size == 0 or conduction.size == 0:
            return {"band_gap": 0.0}

        vbm = valence.max()
        cbm = conduction.min()

        return {"band_gap": cbm - vbm}
    else:
        # 智能获取能带
        return smart_band_gap(eig, pseudo_fermi_level)

def _band_predict(
        model_file_path: Path,
        structure_file_path: Path,
        kpath: str,
        nel_atom: Dict[str, int],
        kmesh: str = None,
        override_overlap: Path = None,
        eig_solver: str = "numpy",
        smearing_method: str = "FD",
        sigma_val: float = None,
        work_path: str = "."
    ):
    """
    使用输入的模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
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
                                          eig_solver=eig_solver,
                                          smearing_method=smearing_method,
                                          sigma_val=sigma_val)

        tbsystem.band.set_kpath(**kpath_config)
        band_data = tbsystem.band.compute(eig_solver=eig_solver)

        import time
        timestamp = int(time.time())
        bandstructure_filename = f"bandstructure_{timestamp}"
        output_bandstructure_path = work_dir / bandstructure_filename
        band_data.export(output_bandstructure_path)

        band_img_filename = f"band_{timestamp}.png"
        output_band_img_path = work_dir / band_img_filename
        band_data.plot(filename=output_band_img_path,
                       emin=-25,emax=8)

    return {"band_structure_file_path": str(output_bandstructure_path) + '.npz',
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
        efermi: float = None,
        pseudo_efermi: float = None,
        julia_script_path: Path = None,
        num_band: int = 30
    ):
    """
    使用输入的模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    assert model_file_path, "模型必须输入"
    assert structure_file_path, "输入的结构必须输入"
    assert not (efermi and pseudo_efermi), "费米能级与粗费米能级不该同时输入！"

    if pseudo_efermi:
        efermi = pseudo_efermi

    if not emin:
        emin=-1e6
    if not emax:
        emax=1e6

    _work_path = Path(generate_work_path()).absolute()
    work_dir = Path(work_path).absolute()

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
                "E_fermi": efermi if efermi else 0,
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
        if efermi is None:
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

        bandstructure_filename = f"bandstructure_{timestamp}.npy"
        output_bandstructure_path = work_dir / bandstructure_filename
        shutil.copy(expected_npy, output_bandstructure_path)

    return {"band_structure_file_path": str(output_bandstructure_path),
            "image_file_path": output_band_img_path,
            "fermi_level": data["E_fermi"]}

def _band_compare(
        dptb_result_path: Path,
        dft_result_path: Path,
        e_min: float = None,
        e_max: float = None,
        match_efermi: bool = False,
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

    # ==========================================================
    # 1. 读取 DFT 能带（ABACUS）
    # ==========================================================
    efermi_dict = _abacus_get_efermi(
        abacus_out_path=dft_result_path
    )
    dft_efermi = efermi_dict["efermi"]

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
    if match_efermi:
        dft_band = dft_band - dft_efermi

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
        dptb_efermi = float(
            np.squeeze(dptb_result["fermi_level"])
        )
    else:
        dptb_efermi = 0.0

    if match_efermi:
        dptb_band = dptb_band - dptb_efermi
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
    if match_efermi:
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
            dft_efermi,
            linestyle="--",
            linewidth=1,
            color="black",
            label="DFT $E_F$"
        )
        plt.axhline(
            dptb_efermi,
            linestyle="--",
            linewidth=1,
            color="red",
            label="DeePTB $E_F$"
        )

    plt.ylim(e_min, e_max)

    plt.xlabel("k-path")
    ylabel = "$E - E_f$ (eV)" if match_efermi else "Energy (eV)"
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

