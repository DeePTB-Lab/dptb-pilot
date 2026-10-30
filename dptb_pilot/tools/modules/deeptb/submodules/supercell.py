import math

import numpy as np
from ase import Atoms


def detect_axis_by_longest_cell(cell):
    """
    Detect the cell-vector index with the largest length.

    Parameters
    ----------
    cell : array-like
        3x3 lattice cell, such as an ASE cell object.

    Returns
    -------
    int
        Index of the longest lattice vector (0, 1, or 2).
    """
    norms = [np.linalg.norm(cell[i]) for i in range(3)]
    axis = int(np.argmax(norms))
    return axis


def build_supercell(system: Atoms, axis='auto', target_length=None, n_repeat=None, verbose=True):
    """
    返回扩展后的系统 (ASE Atoms)。
    axis: 'auto' 或 'x','y','z' 指定沿哪一维 repeat
    target_length: 希望的长度（Å）沿管轴（优先）；若为 None, 使用 n_repeat
    n_repeat: 明确重复次数（覆盖 target_length）
    """
    cell = system.get_cell()
    if axis == 'auto':
        axis = detect_axis_by_longest_cell(cell)
        if verbose:
            print(f"[build] auto-detected tube axis = {axis}")
    else:
        assert axis in ['x','y','z']
        axis = {'x': 0, 'y': 1, 'z': 2}.get(axis, None)

    axis_vec = np.array(cell[axis])
    axis_len = np.linalg.norm(axis_vec)
    if verbose:
        print(f"[build] cell vector length along axis {axis}: {axis_len:.3f} Å")

    if n_repeat is not None:
        rep = int(n_repeat)
    elif target_length is not None:
        rep = int(math.ceil(float(target_length) / axis_len))
    else:
        raise ValueError("Either target_length or n_repeat must be provided")

    if rep < 1:
        rep = 1

    # construct repeat tuple
    repeats = [1, 1, 1]
    repeats[axis] = rep
    supercell = system.repeat(tuple(repeats))
    if verbose:
        final_cell = supercell.get_cell()
        print(
            f"[build] repeated cell {repeats}, final cell lengths: {np.linalg.norm(final_cell[0]):.3f}, {np.linalg.norm(final_cell[1]):.3f}, {np.linalg.norm(final_cell[2]):.3f} Å")
    return supercell, axis, rep


def make_cylinder_indenter(center, radius, height, spacing=1.42, symbol='C'):
    """
    生成近似“固体”圆柱体的原子阵列（沿 z 方向延伸高度）。
    center: (x,y,z) 中心坐标（底面中心）
    radius: 圆柱半径（Å）
    height: 圆柱高度（Å）
    spacing: 原子间近似间距（Å），默认接近石墨间距 ~1.42 Å
    返回 ASE Atoms 对象（符号为 symbol）
    """
    cx, cy, cz = center
    # generate grid in x-y plane that covers cylinder
    # choose grid spacing = spacing / sqrt(2) to ensure near close packing
    dx = spacing * 0.9
    xs = np.arange(cx - radius - dx, cx + radius + dx, dx)
    ys = np.arange(cy - radius - dx, cy + radius + dx, dx)
    zs = np.arange(cz, cz + height + dx, dx)

    coords = []
    for x in xs:
        for y in ys:
            # check inside cylinder cross-section
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                for z in zs:
                    coords.append([x, y, z])

    if len(coords) == 0:
        return Atoms()
    symbols = [symbol] * len(coords)
    ind = Atoms(symbols=symbols, positions=coords)
    return ind


