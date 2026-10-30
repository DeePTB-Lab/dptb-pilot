from pathlib import Path
from typing import Dict, Literal, Optional

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
    override_overlap: Path = None,
    smearing_method: str = "FD",
    sigma_val: float = None,
    work_path: str = "."
) -> BandResult:
    """
    使用输入的SK模型预测结构能带。

    方法选择建议:
        这是普通（非 Julia）能带计算方法。对于结构较小，或费米能级不够准确的情况，
        优先使用本方法，结果更稳妥。只有在确实需要处理大结构时，才考虑
        ``band_predict_with_julia``；Julia 方法虽然速度更快，但只能得到局域能带。

    Parameters
    ----------
    model_file_path : Path
        DeePTB 模型文件路径。
    structure_file_path : Path
        输入结构文件路径，通常为 VASP/POSCAR 格式。
    kpath : str
        K-Path 字符串，例如
        ``"[[0.0,0.0,0.0,50,G],[0.5,0.0,0.5,1,X]]"``。
    nel_atom : Dict[str, int]
        元素到价电子数的映射，例如 ``{"Si": 4, "H": 1}``。
    kmesh : str, optional
        计算费米能级的 k 点网格，例如 ``"[5,5,5]"``；默认
        ``[5,5,5]``。
    override_overlap : Path, optional
        用于覆盖模型所生成 overlap 的文件路径。
    smearing_method : str, optional
        计算费米能级时传给 DeePTB 的展宽方法，默认 ``FD``。
    sigma_val : float, optional
        费米能级展宽宽度；不提供时使用 DeePTB 默认值。
    work_path : str, optional
        能带数据和图片的保存目录。

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
                       override_overlap=override_overlap,
                       smearing_method=smearing_method,
                       sigma_val=sigma_val,
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
        fermi_level: float = None,
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
        元素价电子数字典，例如 ``{"Si": 4}``。未提供 ``fermi_level`` 时用于自动计算费米能级。
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
    fermi_level : float, optional
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
        fermi_level=fermi_level,
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
        match_fermi_level: bool = True,
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
        横轴能量范围。默认使用文件中的能量范围；``match_fermi_level=True`` 时为相对费米能级。
    dos_min, dos_max : float, optional
        纵轴 DOS 范围。
    match_fermi_level : bool, optional
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
        match_fermi_level=match_fermi_level,
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
        fermi_level: float = None,
        pseudo_fermi_level: float = None,
        julia_script_path: Path = None,
        num_band: int = 30
) -> BandResult:
    """
    【重要】非默认能带计算方法：仅在下述适用条件全部满足时使用 Julia 稀疏求解器。
    使用本方法预测并绘制 DeePTB 局域能带。

    重要的适用范围与选择规则:
        Julia 方法只能得到局域能带，但计算速度远快于普通方法。对于结构较小，
        或费米能级不准确的情况，不要使用本方法；此时应使用普通的
        ``band_predict``，结果更保险。

        本方法仅适用于同时满足以下条件的情况：
        1. 结构较大，普通方法的计算成本过高；
        2. 费米能级十分准确（由外部输入，或模型准确度极高）；
        3. 只需要局域能带来获得带隙等信息，而不需要完整能带结构。

        Agent 不应仅因为 Julia 方法更快就默认选择它；如果结构大小或费米能级
        可靠性不明确，应选择普通的 ``band_predict``。

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
    fermi_level : float, optional
        Explicit Fermi level.
    pseudo_fermi_level : float, optional
        Approximate Fermi level. Mutually exclusive with ``fermi_level``.
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
        fermi_level=fermi_level,
        pseudo_fermi_level=pseudo_fermi_level,
        julia_script_path=julia_script_path,
        num_band=num_band,
    )


@mcp.tool()
def band_gap(
        band_structure_file_path: Path,
        fermi_level: Optional[float] = None,
        n_atoms: Optional[int] = None,
        pseudo_fermi_level: Optional[float] = None,
        calculation_method: Literal["auto", "fermi", "smart"] = "auto",
        smart_band_gap_atol: Optional[float] = None,
        smart_band_gap_max_diff: float = 1.0,
) -> BandGapResult:
    """
    Calculate a band gap from a DeePTB band-structure file.

    Input formats are normalized before calculation. Supported files are a
    ``.npz`` archive containing ``eigenvalues``, a trusted ``.npy`` file
    containing a band-data dictionary, and a numeric ``.npy`` eigenvalue array.
    Eigenvalues may have shape ``(Nk, Nband)``, ``(Nband,)``, or
    ``(1, Nk, Nband)``. The last form is treated as a single spin/channel.

    The reference energy is resolved in this order: ``pseudo_fermi_level`` or
    ``fermi_level`` argument, file field ``fermi_level``, file field
    ``E_fermi``, then the legacy ``n_atoms`` estimate. Supplying both explicit
    Fermi-level arguments is an error; numeric zero is always treated as a real
    value.

    Parameters
    ----------
    band_structure_file_path : Path
        Path to a saved DeePTB band-structure ``.npz`` or ``.npy`` file.
    fermi_level : float, optional
        Explicit Fermi level in the same energy reference as ``eigenvalues``.
    n_atoms : int, optional
        Atom count for the legacy electron-count estimate, used only when no
        Fermi level is supplied by an argument or the file.
    pseudo_fermi_level : float, optional
        Approximate reference energy for smart/local-gap estimation. Mutually
        exclusive with ``fermi_level``. In ``auto`` mode, its presence selects
        the smart method.
    calculation_method : {"auto", "fermi", "smart"}, optional
        ``fermi`` finds the nearest occupied and unoccupied states globally.
        ``smart`` searches for a gap near the reference energy and includes
        local-gap validation. ``auto`` uses ``smart`` only when
        ``pseudo_fermi_level`` is explicitly provided; otherwise it uses
        ``fermi``.
    smart_band_gap_atol : float, optional
        Minimum resolved gap for smart search in eV. Defaults to ``1e-3``.
    smart_band_gap_max_diff : float, optional
        Smart-search radius around the reference energy in eV. Defaults to
        ``1.0``.

    Returns
    -------
    BandGapResult
        A stable result with the same keys for both methods. ``band_gap`` equals
        ``gap_width``; ``vbm``/``cbm`` equal ``gap_lower``/``gap_upper``.
        Energies use the input file's reference (they are not shifted by the
        Fermi level). Unavailable values are ``null``. ``status``, ``is_valid``,
        and ``diagnostic_message`` distinguish a valid zero gap from a result
        that could not be determined reliably. ``fermi_level_source`` records
        which argument, file field, or estimate supplied the reference energy;
        ``calculation_method`` records the resolved method rather than ``auto``.
        Band and k-point indices locate the reported edges. ``all_gaps`` is a
        list of ``lower``, ``upper``, and ``width`` mappings; local-search-only
        fields are returned as ``null`` for the conventional method.

    Notes
    -----
    Pickled object-array ``.npy`` files can execute arbitrary code while loading;
    only pass files produced by a trusted DeePTB workflow.
    """
    return _band_gap(
        band_structure_file_path=band_structure_file_path,
        fermi_level=fermi_level,
        n_atoms=n_atoms,
        pseudo_fermi_level=pseudo_fermi_level,
        calculation_method=calculation_method,
        smart_band_gap_atol=smart_band_gap_atol,
        smart_band_gap_max_diff=smart_band_gap_max_diff,
    )


@mcp.tool()
def band_compare(
        dptb_result_path: Path,
        dft_result_path: Path,
        e_min: float = None,
        e_max: float = None,
        match_fermi_level: bool = False,
        delta: float = None,
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
    match_fermi_level : bool, optional
        If true, shift each dataset by its own Fermi level before plotting.
        If false, retain absolute energies and draw both Fermi levels.
    delta : float, optional
        Additional energy shift applied to the DeePTB bands after Fermi-level
        matching. This has an effect only when ``match_fermi_level`` is true.

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
        match_fermi_level=match_fermi_level,
        delta=delta,
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
        model_file_path: 使用的 model 路径。
        structure_file_path: 输入结构文件路径，通常为 VASP/POSCAR 格式。
        k_points: 想要计算的 k 点，格式如 ``"[[0,0,0],[0,0,0.5]]"``。
        override_overlap: 用于覆盖模型所生成 overlap 的文件路径。
        work_path: 哈密顿量和 overlap 文件的保存目录。

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
