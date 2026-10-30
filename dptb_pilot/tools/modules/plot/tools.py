"""General-purpose scientific plotting tools.

All tools deliberately return an absolute PNG path so callers can pass the
result directly to Pilot's artifact/file handling layer.
"""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path
from typing import Any, Sequence

from dptb_pilot.tools.init import mcp


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc


def _output_path(output_dir: str | None, prefix: str) -> Path:
    directory = Path(output_dir or os.environ.get("DPTB_WORK_PATH", os.getcwd()))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{uuid.uuid4().hex}.png"


def _finish(fig: Any, output_dir: str | None, prefix: str) -> str:
    path = _output_path(output_dir, prefix)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()
    return str(path.resolve())


def _series(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


@mcp.tool()
def plot_line(
    x: list[float],
    y: list[float] | list[list[float]],
    labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Create a line plot for one or more scientific series.

    Args:
        x: Shared x coordinates.
        y: Y coordinates, either one series or a list of equal-length series.
        labels: Optional legend labels, one per y series.
        title: Optional figure title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    x_values = _series(x, "x")
    series = y if y and isinstance(y[0], (list, tuple)) else [y]  # type: ignore[index]
    y_values = [_series(item, "y") for item in series]  # type: ignore[arg-type]
    if any(len(item) != len(x_values) for item in y_values):
        raise ValueError("Every y series must have the same length as x")
    if labels is not None and len(labels) != len(y_values):
        raise ValueError("labels must have one entry per y series")
    fig, ax = plt.subplots()
    for index, values in enumerate(y_values):
        ax.plot(x_values, values, label=labels[index] if labels else None)
    if labels:
        ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.25)
    return _finish(fig, output_dir, "line")


@mcp.tool()
def plot_scatter(
    x: list[float],
    y: list[float],
    color: list[float] | None = None,
    size: float = 25.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Create a scientific scatter plot.

    Args:
        x: X coordinates.
        y: Y coordinates.
        color: Optional per-point color values.
        size: Marker area in points squared.
        title: Optional figure title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    x_values, y_values = _series(x, "x"), _series(y, "y")
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same length")
    if size <= 0:
        raise ValueError("size must be positive")
    colors = _series(color, "color") if color is not None else None
    if colors is not None and len(colors) != len(x_values):
        raise ValueError("color must have the same length as x")
    fig, ax = plt.subplots()
    scatter = ax.scatter(x_values, y_values, c=colors, s=size)
    if colors is not None:
        fig.colorbar(scatter, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.25)
    return _finish(fig, output_dir, "scatter")


@mcp.tool()
def plot_bar(
    categories: list[str],
    values: list[float],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Create a categorical bar chart.

    Args:
        categories: Category labels.
        values: Numeric value for each category.
        title: Optional figure title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    y_values = _series(values, "values")
    if len(categories) != len(y_values) or not categories:
        raise ValueError("categories and values must be non-empty and equal in length")
    fig, ax = plt.subplots()
    ax.bar(categories, y_values)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.autofmt_xdate()
    return _finish(fig, output_dir, "bar")


@mcp.tool()
def plot_histogram(
    values: list[float],
    bins: int = 30,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    output_dir: str | None = None,
) -> str:
    """Create a distribution histogram.

    Args:
        values: Samples to bin.
        bins: Number of histogram bins.
        title: Optional figure title.
        xlabel: Optional x-axis label.
        ylabel: Y-axis label.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    samples = _series(values, "values")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    fig, ax = plt.subplots()
    ax.hist(samples, bins=bins)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return _finish(fig, output_dir, "histogram")


@mcp.tool()
def plot_heatmap(
    matrix: list[list[float]],
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_label: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Create a 2D heatmap for matrix-valued scientific data.

    Args:
        matrix: Non-empty rectangular numeric matrix.
        title: Optional figure title.
        xlabel: Optional x-axis label.
        ylabel: Optional y-axis label.
        colorbar_label: Optional colorbar label.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    rows = [_series(row, "matrix row") for row in matrix]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be rectangular")
    fig, ax = plt.subplots()
    image = ax.imshow(rows, aspect="auto", origin="lower")
    cbar = fig.colorbar(image, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return _finish(fig, output_dir, "heatmap")


@mcp.tool()
def plot_subplots(
    plots: list[dict[str, Any]],
    columns: int = 2,
    title: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Create a multi-panel figure from line or scatter plot definitions.

    Args:
        plots: Plot definitions with ``kind`` (``line`` or ``scatter``), ``x``,
            ``y``, and optional ``title``, ``xlabel``, and ``ylabel`` fields.
        columns: Number of subplot columns.
        title: Optional overall figure title.
        output_dir: Optional directory for the generated PNG.
    Returns:
        Absolute path to the generated PNG image.
    """
    plt = _matplotlib()
    if not plots:
        raise ValueError("plots must not be empty")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    rows = math.ceil(len(plots) / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False)
    flat_axes = [axis for row in axes for axis in row]
    for axis, spec in zip(flat_axes, plots):
        kind = spec.get("kind", "line")
        x_values = _series(spec.get("x", []), "x")
        y_values = _series(spec.get("y", []), "y")
        if len(x_values) != len(y_values):
            raise ValueError("Each subplot x and y must have equal lengths")
        if kind == "line":
            axis.plot(x_values, y_values)
        elif kind == "scatter":
            axis.scatter(x_values, y_values)
        else:
            raise ValueError("subplot kind must be 'line' or 'scatter'")
        axis.set(title=spec.get("title"), xlabel=spec.get("xlabel"), ylabel=spec.get("ylabel"))
        axis.grid(True, alpha=0.2)
    for axis in flat_axes[len(plots):]:
        axis.remove()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, output_dir, "subplots")
