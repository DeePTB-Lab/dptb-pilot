from pathlib import Path

import pytest

from dptb_pilot.tools.modules.plot import tools


@pytest.mark.parametrize(
    "function,args,prefix",
    [
        (tools.plot_line, ([0, 1], [1, 2]), "line_"),
        (tools.plot_scatter, ([0, 1], [1, 2]), "scatter_"),
        (tools.plot_bar, (["a", "b"], [1, 2]), "bar_"),
        (tools.plot_histogram, ([1, 2, 2, 3],), "histogram_"),
        (tools.plot_heatmap, ([[1, 2], [3, 4]],), "heatmap_"),
        (tools.plot_subplots, ([{"kind": "line", "x": [0, 1], "y": [1, 2]}],), "subplots_"),
    ],
)
def test_plot_tools_return_png_path(tmp_path, function, args, prefix):
    path = Path(function(*args, output_dir=str(tmp_path)))
    assert path.is_absolute()
    assert path.name.startswith(prefix)
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_plot_line_rejects_mismatched_series(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        tools.plot_line([0, 1], [1], output_dir=str(tmp_path))
