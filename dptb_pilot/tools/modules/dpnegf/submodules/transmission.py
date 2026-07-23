import time
from pathlib import Path
from typing import Any, Dict

import torch
from matplotlib import pyplot as plt


def _to_1d_list(value: Any) -> list:
    """Convert a tensor/array-like value to a 1D Python list for plotting."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        return value.reshape(-1).tolist()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _sort_by_energy(energy: list, transmission: list) -> tuple[list, list]:
    """Sort energy values and keep transmission values aligned."""
    sorted_indices = sorted(range(len(energy)), key=lambda idx: float(energy[idx]))
    return (
        [energy[idx] for idx in sorted_indices],
        [transmission[idx] for idx in sorted_indices],
    )


def _annotation_layout(energy: list) -> list[tuple[int, int, str, str]]:
    """Spread dense point labels across offset columns and rows."""
    if not energy:
        return []

    x_values = [float(value) for value in energy]
    x_span = max(x_values) - min(x_values)
    close_threshold = x_span * 0.035 if x_span else 0.0
    column_offsets = [10, -10, 120, -120, 230, -230]
    max_rows_per_column = 7
    row_step = 14
    offsets = [(8, 8, "left", "bottom") for _ in x_values]

    cluster_start = 0
    while cluster_start < len(x_values):
        cluster_end = cluster_start + 1
        while (
                cluster_end < len(x_values)
                and x_values[cluster_end] - x_values[cluster_end - 1] <= close_threshold
        ):
            cluster_end += 1

        cluster_size = cluster_end - cluster_start
        if cluster_size == 1:
            x_offset = 8 if cluster_start % 2 == 0 else -8
            y_offset = 8 if cluster_start % 3 != 1 else -10
            offsets[cluster_start] = (
                x_offset,
                y_offset,
                "left" if x_offset > 0 else "right",
                "bottom" if y_offset > 0 else "top",
            )
        else:
            column_count = min(
                len(column_offsets),
                max(2, (cluster_size + max_rows_per_column - 1) // max_rows_per_column),
            )
            row_count = (cluster_size + column_count - 1) // column_count
            for local_idx, point_idx in enumerate(range(cluster_start, cluster_end)):
                x_offset = column_offsets[local_idx % column_count]
                row_idx = local_idx // column_count
                y_offset = int(round((row_idx - (row_count - 1) / 2) * row_step))
                y_offset += 8 if y_offset >= 0 else -8
                offsets[point_idx] = (
                    x_offset,
                    y_offset,
                    "left" if x_offset > 0 else "right",
                    "bottom" if y_offset > 0 else "top",
                )

        cluster_start = cluster_end

    return offsets


def _transmission_plot(
        negf_out_path: Path,
        label: bool = False,
        work_path: str = ".",
        image_name: str = None,
) -> Dict[str, str]:
    """
    从 DPNEGF ``negf.out.pth`` 文件绘制透射谱。

    Parameters
    ----------
    negf_out_path : Path
        DPNEGF 输出文件路径，例如 ``negf.out.pth``。文件中需要包含
        ``uni_grid`` 和 ``T_avg``。
    label : bool, optional
        是否在每个数据点附近标注 ``(x, y)`` 数值。
    work_path : str, optional
        图片保存目录。
    image_name : str, optional
        输出图片文件名。默认使用时间戳生成 ``transmission_plot_*.png``。

    Returns
    -------
    dict
        包含生成的透射谱图片路径。
    """

    negf_out_path = Path(negf_out_path).absolute()
    work_dir = Path(work_path).absolute()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not negf_out_path.exists():
        raise FileNotFoundError(f"NEGF output file not found: {negf_out_path}")

    negf_out = torch.load(negf_out_path, map_location="cpu", weights_only=False)
    if "uni_grid" not in negf_out or "T_avg" not in negf_out:
        raise KeyError("NEGF 输出文件中必须包含 'uni_grid' 和 'T_avg'")

    energy = _to_1d_list(negf_out["uni_grid"])
    transmission = _to_1d_list(negf_out["T_avg"])
    if len(energy) != len(transmission):
        raise ValueError("'uni_grid' 和 'T_avg' 的数据点数量不一致")
    energy, transmission = _sort_by_energy(energy, transmission)

    fig, ax = plt.subplots(figsize=(7 if label else 6, 5))
    ax.plot(
        energy,
        transmission,
        color="black",
        linewidth=1,
        marker="o",
        markersize=4,
        label="Transmission",
    )

    if label:
        for x_value, y_value, (x_offset, y_offset, ha, va) in zip(
                energy,
                transmission,
                _annotation_layout(energy),
        ):
            ax.annotate(
                f"({x_value:.4g}, {y_value:.4g})",
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                fontsize=7,
                ha=ha,
                va=va,
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.45",
                    "linewidth": 0.4,
                    "shrinkA": 0,
                    "shrinkB": 2,
                },
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )
        ax.margins(x=0.1, y=0.18)
    else:
        ax.margins(x=0.05, y=0.08)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Transmission")
    ax.set_title("Transmission vs Energy")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    else:
        ax.legend(loc="best")
    fig.tight_layout()

    if image_name is None:
        image_name = f"transmission_plot_{int(time.time())}.png"
    transmission_img_path = work_dir / image_name
    fig.savefig(transmission_img_path, dpi=300)
    plt.close(fig)

    return {"transmission_plot_path": str(transmission_img_path)}
