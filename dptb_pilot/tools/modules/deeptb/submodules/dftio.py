"""Input normalization and execution helpers for :mod:`dftio`."""

from __future__ import annotations

import os
import re
import shutil
import subprocess as sp
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Union

from dptb_pilot.tools.modules.util.comm import generate_work_path

PathInput = Union[str, os.PathLike[str]]
InputCollection = Union[PathInput, Sequence[PathInput]]


def _as_path_list(work_root: InputCollection) -> list[Path]:
    """Convert public path forms to validated absolute directories."""
    if work_root is None:
        raise ValueError("work_root must contain at least one directory")
    if isinstance(work_root, (str, os.PathLike)):
        values: Iterable[PathInput] = [work_root]
    elif isinstance(work_root, Sequence):
        values = work_root
    else:
        raise TypeError("work_root must be a directory path or a sequence of directory paths")

    paths: list[Path] = []
    for value in values:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"work_root contains an invalid path: {value!r}")
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"DFT input directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"DFT input path is not a directory: {path}")
        if path not in paths:
            paths.append(path)
    if not paths:
        raise ValueError("work_root must contain at least one directory")
    return paths


def _prefix_hints(prefix: Any) -> list[str]:
    if prefix is None:
        return []
    if isinstance(prefix, str):
        return [prefix] if prefix.strip() else []
    if isinstance(prefix, Sequence):
        return [str(item) for item in prefix if str(item).strip()]
    return [str(prefix)]


def _looks_like_calculation(path: Path, mode: str) -> bool:
    """Best-effort detection of a calculation directory."""
    mode = mode.lower()
    if mode == "abacus":
        return (
            any(item.is_dir() and item.name.upper().startswith("OUT.") for item in path.iterdir())
            or ((path / "INPUT").is_file() and any(path.glob("running_*.log")))
            or path.name.upper().startswith("OUT.")
        )
    if mode == "vasp":
        return (path / "POSCAR").is_file() or (path / "OUTCAR").is_file()
    markers = {"rescu": ("input",), "siesta": ("*.fdf", "*.out"),
               "gaussian": ("*.gjf", "*.log"), "pyatb": ("input",)}.get(mode, ())
    return any(path.glob(marker) for marker in markers)


def _expand_input_paths(paths: list[Path], prefix: Any, mode: str) -> list[Path]:
    """Expand dataset roots and retain deterministic source order."""
    hints = _prefix_hints(prefix)
    expanded: list[Path] = []
    for path in paths:
        children = sorted((item for item in path.iterdir() if item.is_dir()), key=lambda p: p.name.lower())
        if _looks_like_calculation(path, mode) or not children:
            candidates = [path]
        else:
            hinted = [child for child in children if any(hint in child.name for hint in hints)]
            marked = [child for child in children if _looks_like_calculation(child, mode)]
            candidates = hinted or marked or children
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in expanded:
                expanded.append(candidate)
    if not expanded:
        raise ValueError("No DFT calculation directories were found in work_root")
    return expanded


def _safe_output_name(output_dir_name: Union[str, os.PathLike[str]]) -> str:
    output_name = Path(output_dir_name)
    if output_name.is_absolute() or ".." in output_name.parts:
        raise ValueError("output_dir_name must be a relative directory name")
    if not output_name.name or output_name.name in {".", ".."}:
        raise ValueError("output_dir_name must name a directory")
    return str(output_name)


def _copy_input(source: Path, destination: Path, mode: str) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    # An OUT.* directory is also accepted as a direct ABACUS input.  Keep it
    # nested because AbacusParser resolves OUT.* relative to its calculation root.
    if mode.lower() == "abacus" and source.name.upper().startswith("OUT."):
        shutil.copytree(source, destination / source.name, symlinks=True)
        return
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target, follow_symlinks=False)


def _dftio_executable() -> list[str]:
    executable = shutil.which("dftio") or shutil.which("dftio.exe")
    return [executable] if executable else [sys.executable, "-m", "dftio"]


def _map_output_dirs(source_paths: list[Path], output_path: Path) -> dict[str, str]:
    """Map dftio's ``formula.index`` directories back to source order."""
    parsed_dirs = [item for item in output_path.iterdir() if item.is_dir()]
    indexed: dict[int, Path] = {}
    for item in parsed_dirs:
        match = re.search(r"\.(\d+)$", item.name)
        if match:
            indexed[int(match.group(1))] = item
    ordered = [indexed[index] for index in range(len(source_paths)) if index in indexed]
    if len(ordered) != len(source_paths):
        ordered = sorted(parsed_dirs, key=lambda p: p.name)
    return {str(source): str(ordered[index]) for index, source in enumerate(source_paths)
            if index < len(ordered)}


def _dftio_parse_abacus(
    work_root: InputCollection, prefix: Any = "abacus", format: str = "dat",
    output_dir_name: str = "parse_result", out_hamiltonian: bool = False,
    out_overlap: bool = False, out_density_matrix: bool = False,
    out_eigenvalue: bool = False,
):
    """Parse ABACUS inputs through the installed Python dftio parser."""
    from dftio.io.abacus.abacus_parser import AbacusParser
    from tqdm import tqdm

    paths = _expand_input_paths(_as_path_list(work_root), prefix, "abacus")
    parser = AbacusParser(root=[str(path) for path in paths], prefix="")
    output_path = Path(generate_work_path()).absolute() / _safe_output_name(output_dir_name)
    for idx in tqdm(range(len(parser))):
        parser.write(idx=idx, format=format, hamiltonian=out_hamiltonian,
                     overlap=out_overlap, outroot=output_path, eigenvalue=out_eigenvalue,
                     density_matrix=out_density_matrix, band_index_min=0)
    return {"output_path": output_path, "input_mapping": _map_output_dirs(paths, output_path)}


def _dftio_parse(
    work_root: InputCollection, mode: str = "abacus", prefix: Any = "abacus",
    output_dir_name: str = "parse_result", out_hamiltonian: bool = False,
    out_overlap: bool = False, out_density_matrix: bool = False,
    out_eigenvalue: bool = False,
):
    """Normalize one or more DFT directories and run ``dftio parse``.

    ``work_root`` may be a dataset root, one calculation directory, or a
    sequence of calculation directories.  ``input_mapping`` maps every source
    directory to the corresponding parsed directory when available.
    """
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("mode must be a non-empty string")
    mode = mode.strip().lower()
    output_name = _safe_output_name(output_dir_name)
    source_paths = _expand_input_paths(_as_path_list(work_root), prefix, mode)
    work_path = Path(generate_work_path()).absolute()
    work_path.mkdir(parents=True, exist_ok=True)

    canonical_prefix = "dftio_input"
    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)
        normalized_root = temp_path / "inputs"
        normalized_root.mkdir()
        for index, source in enumerate(source_paths):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name).strip("._") or "input"
            _copy_input(source, normalized_root / f"{canonical_prefix}.{index:06d}.{safe_name}", mode)

        parsed_root = temp_path / "parsed"
        cmd = [*_dftio_executable(), "parse", "-m", mode, "-r", str(normalized_root),
               "-p", canonical_prefix, "-o", str(parsed_root)]
        if out_hamiltonian:
            cmd.append("-ham")
        if out_overlap:
            cmd.append("-ovp")
        if out_density_matrix:
            cmd.append("-dm")
        if out_eigenvalue:
            cmd.append("-eig")
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"dftio execution failed (exit code {result.returncode}):\n{details}")
        if not parsed_root.is_dir():
            raise RuntimeError("dftio completed without creating an output directory")

        output_path = work_path / output_name
        shutil.copytree(parsed_root, output_path, dirs_exist_ok=True)
        mapping = _map_output_dirs(source_paths, output_path)

    return {"output_path": output_path, "input_mapping": mapping}
