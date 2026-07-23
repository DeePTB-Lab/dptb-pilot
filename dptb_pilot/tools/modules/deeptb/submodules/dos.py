import ast
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from dptb.postprocess.unified import TBSystem


def _parse_sequence(value, default=None):
    if value is None:
        return default
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _dos_predict(
        model_file_path: Path,
        structure_file_path: Path,
        nel_atom: Optional[Dict[str, int]] = None,
        kmesh: Union[str, list, tuple] = None,
        erange: Union[str, list, tuple] = None,
        npts: int = 100,
        smearing: str = "gaussian",
        sigma: float = 0.05,
        pdos: bool = False,
        is_gamma_center: bool = True,
        override_overlap: Path = None,
        efermi: float = None,
        fermi_kmesh: Union[str, list, tuple] = None,
        fermi_smearing_method: str = "FD",
        work_path: str = ".",
        plot_pdos: bool = False,
        selected_orbitals: Union[str, list] = None,
        xlim: Union[str, list, tuple] = None,
):
    """
    使用 DeePTB ``TBSystem`` 计算结构的 DOS/PDOS。

    Parameters
    ----------
    model_file_path : Path
        DeePTB 模型文件路径。
    structure_file_path : Path
        输入结构文件路径，通常为 VASP/POSCAR 格式。
    nel_atom : dict, optional
        元素价电子数字典，例如 ``{"Si": 4}``。当未显式提供 ``efermi`` 时用于计算费米能级。
    kmesh : str or list, optional
        DOS 计算使用的 k 点网格，例如 ``"[5,5,5]"``。默认 ``[5, 5, 5]``。
    erange : str or list, optional
        相对费米能级的能量范围，例如 ``"[-10,10]"``。默认 ``[-10, 10]``。
    npts : int, optional
        能量网格点数。
    smearing : str, optional
        DOS 展宽方式，支持 ``gaussian`` 和 ``lorentzian``。
    sigma : float, optional
        展宽宽度，单位 eV。
    pdos : bool, optional
        是否计算投影 DOS。
    is_gamma_center : bool, optional
        是否使用 Gamma-centered k 点网格。
    override_overlap : Path, optional
        用于覆盖模型 overlap 的文件。
    efermi : float, optional
        外部指定费米能级；提供后不会自动计算。
    fermi_kmesh : str or list, optional
        自动计算费米能级时使用的 k 点网格，默认与 ``kmesh`` 相同。
    fermi_smearing_method : str, optional
        自动计算费米能级时使用的 smearing 方法。
    work_path : str, optional
        DOS 数据和图片保存目录。
    plot_pdos : bool, optional
        绘图时是否同时画 PDOS。
    selected_orbitals : str or list, optional
        绘制 PDOS 时选中的轨道下标或标签列表。
    xlim : str or list, optional
        绘图横轴范围。

    Returns
    -------
    dict
        包含 DOS 数据文件、图片文件和费米能级的路径/数值。
    """

    assert model_file_path, "模型必须输入"
    assert structure_file_path, "输入的结构必须输入"

    model_file_path = Path(model_file_path).absolute()
    structure_file_path = Path(structure_file_path).absolute()
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not structure_file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_file_path}")

    kmesh = _parse_sequence(kmesh, [5, 5, 5])
    erange = _parse_sequence(erange, [-10, 10])
    fermi_kmesh = _parse_sequence(fermi_kmesh, kmesh)
    selected_orbitals = _parse_sequence(selected_orbitals, selected_orbitals)
    xlim = _parse_sequence(xlim, None)

    tbsystem = TBSystem(
        data=str(structure_file_path),
        calculator=str(model_file_path),
        override_overlap=str(override_overlap) if override_overlap else None,
    )

    if efermi is not None:
        tbsystem.set_efermi(efermi)
    elif nel_atom is not None:
        tbsystem.set_electrons(nel_atom=nel_atom)
        efermi = tbsystem.get_efermi(
            kmesh=fermi_kmesh,
            is_gamma_center=is_gamma_center,
            smearing_method=fermi_smearing_method,
        )
    else:
        efermi = 0.0

    dos_accessor = tbsystem.get_dos(
        kmesh=kmesh,
        is_gamma_center=is_gamma_center,
        erange=erange,
        npts=npts,
        smearing=smearing,
        sigma=sigma,
        pdos=pdos,
    )

    timestamp = int(time.time())
    dos_data_path = work_dir / f"dos_{timestamp}.npz"
    dos_image_path = work_dir / f"dos_{timestamp}.png"

    dos_accessor.save(str(dos_data_path))
    dos_accessor.plot(
        filename=str(dos_image_path),
        show=False,
        xlim=xlim,
        plot_pdos=plot_pdos,
        selected_orbitals=selected_orbitals,
    )

    return {
        "dos_file_path": str(dos_data_path),
        "image_file_path": str(dos_image_path),
        "fermi_level": float(efermi),
    }


def _dos_plot(
        dos_file_path: Path,
        e_min: float = None,
        e_max: float = None,
        dos_min: float = None,
        dos_max: float = None,
        match_efermi: bool = True,
        plot_pdos: bool = False,
        selected_orbitals: Union[str, list] = None,
        work_path: str = ".",
):
    """
    从 DeePTB DOS ``.npz`` 文件绘制 DOS/PDOS 图。

    Parameters
    ----------
    dos_file_path : Path
        ``dos_predict`` 生成的 DOS ``.npz`` 文件，需包含 ``energy`` 和 ``dos``。
    e_min, e_max : float, optional
        横轴能量范围。默认使用文件中的能量范围；``match_efermi=True`` 时为相对费米能级。
    dos_min, dos_max : float, optional
        纵轴 DOS 范围。
    match_efermi : bool, optional
        是否将能量减去文件中的 ``fermi_level``，默认画 ``E - E_F``。
    plot_pdos : bool, optional
        文件包含 ``pdos`` 时是否同时绘制 PDOS。
    selected_orbitals : str or list, optional
        绘制 PDOS 时选中的轨道下标或标签列表，例如 ``"[0,1]"``。
    work_path : str, optional
        图片保存目录。

    Returns
    -------
    dict
        包含生成的 DOS 图片路径。
    """

    dos_file_path = Path(dos_file_path).absolute()
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not dos_file_path.exists():
        raise FileNotFoundError(f"DOS file not found: {dos_file_path}")

    data = np.load(dos_file_path, allow_pickle=True)
    if "energy" not in data or "dos" not in data:
        raise KeyError("DOS 文件中必须包含 'energy' 和 'dos'")

    energy = data["energy"]
    dos = data["dos"]
    efermi = float(np.squeeze(data["fermi_level"])) if "fermi_level" in data else 0.0

    x_energy = energy - efermi if match_efermi else energy
    selected_orbitals = _parse_sequence(selected_orbitals, selected_orbitals)

    plt.figure(figsize=(6, 5))
    plt.plot(x_energy, dos, color="black", linewidth=1, label="Total DOS")

    if plot_pdos and "pdos" in data:
        pdos = data["pdos"]
        labels = data["pdos_labels"].tolist() if "pdos_labels" in data else None

        if selected_orbitals is None:
            indices = range(pdos.shape[1])
        else:
            indices = []
            for item in selected_orbitals:
                if isinstance(item, int):
                    indices.append(item)
                elif isinstance(item, str) and labels:
                    if item not in labels:
                        raise ValueError(f"PDOS label not found: {item}")
                    indices.append(labels.index(item))

        for i in indices:
            label = labels[i] if labels else f"Orbital {i}"
            plt.plot(x_energy, pdos[:, i], linewidth=0.8, alpha=0.8, label=label)

        plt.legend(fontsize=8)

    if match_efermi:
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Energy - $E_F$ (eV)")
    else:
        plt.axvline(efermi, linestyle="--", linewidth=1)
        plt.xlabel("Energy (eV)")

    if e_min is None:
        e_min = float(np.min(x_energy))
    if e_max is None:
        e_max = float(np.max(x_energy))
    plt.xlim(e_min, e_max)

    if dos_min is None:
        dos_min = 0.0
    if dos_max is not None:
        plt.ylim(dos_min, dos_max)
    else:
        plt.ylim(bottom=dos_min)

    plt.ylabel("DOS (states/eV)")
    plt.title("Density of States")

    timestamp = int(time.time())
    dos_img_path = work_dir / f"dos_plot_{timestamp}.png"

    plt.tight_layout()
    plt.savefig(dos_img_path, dpi=300)
    plt.close()

    return {"dos_plot_path": str(dos_img_path)}
