from pathlib import Path
from typing import Dict

from dptb_pilot.tools.init import mcp
from dptb_pilot.tools.modules.deeptb.results_unified import (
    BandCompareResult,
    BandGapResult,
    BandResult,
    DosPlotResult,
    DosResult,
    HamiltonianResult,
)
from dptb_pilot.tools.modules.deeptb.submodules.band import (
    _band_compare,
    _band_gap,
    _band_predict,
    _band_predict_with_julia,
)
from dptb_pilot.tools.modules.deeptb.submodules.dos import _dos_plot, _dos_predict
from dptb_pilot.tools.modules.deeptb.submodules.hamiltonian import _hamiltonian_predict

@mcp.tool()
def band_predict(
    model_file_path: Path,
    structure_file_path: Path,
    kpath: str,
    nel_atom: Dict[str, int],
    kmesh: str = None,
    work_path: str = "."
) -> BandResult:
    """
    使用输入的SK模型预测结构能带。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        kpath: K-Path，格式如"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"
        nel_atom: Dictionary mapping element symbols to number of valence electrons. Example: {'Si': 4, 'H': 1}
        kmesh: 用于计算费米能级的k点网格，默认为[5,5,5]，格式如"[5,5,5]"
        work_path: 能带信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    return _band_predict(model_file_path=model_file_path,
                       structure_file_path=structure_file_path,
                       kpath=kpath,
                       nel_atom=nel_atom,
                       kmesh=kmesh,
                       work_path=work_path)

@mcp.tool()
def dos_predict(
        model_file_path: Path,
        structure_file_path: Path,
        nel_atom: Dict[str, int] = None,
        kmesh: str = None,
        erange: str = None,
        npts: int = 100,
        smearing: str = "gaussian",
        sigma: float = 0.05,
        pdos: bool = False,
        is_gamma_center: bool = True,
        override_overlap: Path = None,
        efermi: float = None,
        fermi_kmesh: str = None,
        fermi_smearing_method: str = "FD",
        work_path: str = ".",
        plot_pdos: bool = False,
        selected_orbitals: str = None,
        xlim: str = None,
) -> DosResult:
    """
    使用 DeePTB TBSystem 计算结构的 DOS/PDOS 并保存数据与图片。

    Parameters
    ----------
    model_file_path : Path
        DeePTB 模型文件路径。
    structure_file_path : Path
        输入结构文件路径，通常为 VASP/POSCAR 格式。
    nel_atom : Dict[str, int], optional
        元素价电子数字典，例如 ``{"Si": 4}``。未提供 ``efermi`` 时用于自动计算费米能级。
    kmesh : str, optional
        DOS 计算 k 点网格，例如 ``"[5,5,5]"``，默认 ``[5,5,5]``。
    erange : str, optional
        相对费米能级的能量范围，例如 ``"[-10,10]"``，默认 ``[-10,10]``。
    npts : int, optional
        能量网格点数。
    smearing : str, optional
        展宽方式，支持 ``gaussian`` 和 ``lorentzian``。
    sigma : float, optional
        展宽宽度，单位 eV。
    pdos : bool, optional
        是否计算投影 DOS。
    is_gamma_center : bool, optional
        是否使用 Gamma-centered k 点网格。
    override_overlap : Path, optional
        用于覆盖模型 overlap 的文件。
    efermi : float, optional
        外部指定费米能级。
    fermi_kmesh : str, optional
        自动计算费米能级时使用的 k 点网格，默认与 ``kmesh`` 相同。
    fermi_smearing_method : str, optional
        自动计算费米能级时使用的 smearing 方法。
    work_path : str, optional
        DOS 数据和图片保存目录。
    plot_pdos : bool, optional
        绘图时是否同时画 PDOS。
    selected_orbitals : str, optional
        绘制 PDOS 时选中的轨道下标或标签列表，例如 ``"[0,1]"``。
    xlim : str, optional
        绘图横轴范围，例如 ``"[-5,5]"``。

    Returns
    -------
    DosResult
        DOS 数据文件路径、图片路径和费米能级。
    """
    return _dos_predict(
        model_file_path=model_file_path,
        structure_file_path=structure_file_path,
        nel_atom=nel_atom,
        kmesh=kmesh,
        erange=erange,
        npts=npts,
        smearing=smearing,
        sigma=sigma,
        pdos=pdos,
        is_gamma_center=is_gamma_center,
        override_overlap=override_overlap,
        efermi=efermi,
        fermi_kmesh=fermi_kmesh,
        fermi_smearing_method=fermi_smearing_method,
        work_path=work_path,
        plot_pdos=plot_pdos,
        selected_orbitals=selected_orbitals,
        xlim=xlim,
    )

@mcp.tool()
def dos_plot(
        dos_file_path: Path,
        e_min: float = None,
        e_max: float = None,
        dos_min: float = None,
        dos_max: float = None,
        match_efermi: bool = True,
        plot_pdos: bool = False,
        selected_orbitals: str = None,
        work_path: str = ".",
) -> DosPlotResult:
    """
    从 DeePTB DOS ``.npz`` 文件绘制 DOS/PDOS 图。

    Parameters
    ----------
    dos_file_path : Path
        ``dos_predict`` 生成或同格式的 DOS ``.npz`` 文件，包含 ``energy`` 和 ``dos``。
    e_min, e_max : float, optional
        横轴能量范围。默认使用文件中的能量范围；``match_efermi=True`` 时为相对费米能级。
    dos_min, dos_max : float, optional
        纵轴 DOS 范围。
    match_efermi : bool, optional
        是否将能量减去文件中的 ``fermi_level``，默认画 ``E - E_F``。
    plot_pdos : bool, optional
        文件包含 ``pdos`` 时是否同时绘制 PDOS。
    selected_orbitals : str, optional
        绘制 PDOS 时选中的轨道下标或标签列表，例如 ``"[0,1]"``。
    work_path : str, optional
        图片保存目录。

    Returns
    -------
    DosPlotResult
        生成的 DOS 图片路径。
    """
    return _dos_plot(
        dos_file_path=dos_file_path,
        e_min=e_min,
        e_max=e_max,
        dos_min=dos_min,
        dos_max=dos_max,
        match_efermi=match_efermi,
        plot_pdos=plot_pdos,
        selected_orbitals=selected_orbitals,
        work_path=work_path,
    )


@mcp.tool()
def band_predict_with_julia(
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
) -> BandResult:
    """
    Predict and plot a DeePTB band structure using the Julia sparse solver workflow.

    Parameters
    ----------
    model_file_path : Path
        DeePTB model file.
    structure_file_path : Path
        Input structure file in VASP/POSCAR format.
    kpath : str
        K-path string such as ``[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]``.
    nel_atom : Dict[str, int]
        Valence electron counts by element, e.g. ``{"Si": 4}``.
    kmesh : str, optional
        K-point mesh string used when needed, e.g. ``[5,5,5]``.
    override_overlap : Path, optional
        Overlap file used to override model-generated overlap.
    eig_solver : str, optional
        Eigenvalue solver label retained for compatibility with the Python workflow.
    work_path : str, optional
        Directory where band data and image are saved.
    emin, emax : float, optional
        Energy window for plotting.
    efermi : float, optional
        Explicit Fermi level.
    pseudo_efermi : float, optional
        Approximate Fermi level. Mutually exclusive with ``efermi``.
    julia_script_path : Path, optional
        Path to the Julia sparse solver script. If omitted, DeePTB's bundled script
        is used.
    num_band : int, optional
        Number of bands requested from the sparse solver.

    Returns
    -------
    BandResult
        Paths to the generated band-structure file and plot plus the Fermi level.
    """
    return _band_predict_with_julia(
        model_file_path=model_file_path,
        structure_file_path=structure_file_path,
        kpath=kpath,
        nel_atom=nel_atom,
        kmesh=kmesh,
        override_overlap=override_overlap,
        eig_solver=eig_solver,
        work_path=work_path,
        emin=emin,
        emax=emax,
        efermi=efermi,
        pseudo_efermi=pseudo_efermi,
        julia_script_path=julia_script_path,
        num_band=num_band,
    )


@mcp.tool()
def band_gap(
        band_structure_file_path: Path,
        fermi_level: float = None,
        n_atoms: int = None,
        pseudo_fermi_level: float = None
) -> BandGapResult:
    """
    Calculate the band gap from a DeePTB band-structure file.

    Parameters
    ----------
    band_structure_file_path : Path
        Path to a saved DeePTB band-structure ``.npz`` or ``.npy`` file.
    fermi_level : float, optional
        Explicit Fermi level used to split occupied/unoccupied states.
    n_atoms : int, optional
        Number of atoms for the fallback Fermi-level estimator.
    pseudo_fermi_level : float, optional
        Approximate Fermi level for band-center based gap estimation.

    Returns
    -------
    BandGapResult
        Band gap and optional VBM/CBM details.
    """
    return _band_gap(
        band_structure_file_path=band_structure_file_path,
        fermi_level=fermi_level,
        n_atoms=n_atoms,
        pseudo_fermi_level=pseudo_fermi_level,
    )


@mcp.tool()
def band_compare(
        dptb_result_path: Path,
        dft_result_path: Path,
        e_min: float = None,
        e_max: float = None
) -> BandCompareResult:
    """
    Plot a DeePTB-vs-DFT band-structure comparison.

    Parameters
    ----------
    dptb_result_path : Path
        Path to a DeePTB band-structure result file.
    dft_result_path : Path
        Path to an ABACUS output directory containing ``BANDS_1.dat`` and Fermi
        level logs.
    e_min : float, optional
        Lower energy bound relative to Fermi level.
    e_max : float, optional
        Upper energy bound relative to Fermi level.

    Returns
    -------
    BandCompareResult
        Dictionary with ``band_compare_path`` pointing to the generated image.
    """
    return _band_compare(
        dptb_result_path=dptb_result_path,
        dft_result_path=dft_result_path,
        e_min=e_min,
        e_max=e_max,
    )


@mcp.tool()
def hamiltonian_predict(
        model_file_path: Path,
        structure_file_path: Path,
        k_points: str,
        override_overlap: Path = None,
        work_path: str = "."
) -> HamiltonianResult:
    """
    使用输入的SK模型预测结构哈密顿量。

    参数:
        model_file_name: 使用的model路径。
        structure_file_name: 输入的结构文件路径。结构文件应为vasp的格式
        k_points: 想要计算的k点，格式如"[[0,0,0],[0,0,0.5]]"
        override_overlap: 覆盖的overlap文件，使用后覆盖模型产生的overlap
        work_path: 哈密顿量信息的保存路径。注意应该是文件夹而不是文件。

    返回:
        包含能带文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    return _hamiltonian_predict(model_file_path=model_file_path,
                              structure_file_path=structure_file_path,
                              k_points=k_points,
                              override_overlap=override_overlap,
                              work_path=work_path)