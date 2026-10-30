import re
import shlex
import shutil
import subprocess as sp
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from dptb_pilot.tools.modules.util.comm import generate_work_path


_STRU_SECTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _copy_extra_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def _reference_basename(reference: str) -> str:
    return reference.replace("\\", "/").rsplit("/", 1)[-1]


def _assign_reference_paths(
    references: List[str],
    paths: List[Path],
    *,
    orbital: bool,
    labels: Optional[List[str]] = None,
) -> List[Optional[Path]]:
    """Match basenames and element labels before falling back to input order."""
    candidates = [path for path in paths if (path.suffix.lower() == ".orb") == orbital]
    assignments: List[Optional[Path]] = [None] * len(references)
    unused = list(candidates)

    for index, reference in enumerate(references):
        basename = _reference_basename(reference).casefold()
        match = next(
            (path for path in unused if path.name.casefold() == basename), None
        )
        if match is not None:
            assignments[index] = match
            unused.remove(match)

    if labels is not None:
        for index, (assignment, label) in enumerate(zip(assignments, labels)):
            if assignment is not None:
                continue
            label_pattern = re.compile(
                rf"^{re.escape(label)}(?:[^A-Za-z0-9]|$)", re.IGNORECASE
            )
            match = next(
                (path for path in unused if label_pattern.match(path.name)), None
            )
            if match is not None:
                assignments[index] = match
                unused.remove(match)

    for index, assignment in enumerate(assignments):
        if assignment is None and unused:
            assignments[index] = unused.pop(0)

    return assignments


def _update_stru_file_references(stru_path: Path, pp_orb_paths: List[Path]) -> None:
    if not pp_orb_paths:
        return

    lines = stru_path.read_text(encoding="utf-8").splitlines(keepends=True)
    pp_entries: List[tuple[int, str, str]] = []
    orbital_entries: List[tuple[int, str]] = []
    section: Optional[str] = None

    for index, line in enumerate(lines):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        if _STRU_SECTION_PATTERN.fullmatch(content):
            section = content
            continue

        tokens = content.split()
        if section == "ATOMIC_SPECIES" and len(tokens) >= 3:
            pp_entries.append((index, tokens[2], tokens[0]))
        elif section == "NUMERICAL_ORBITAL" and tokens:
            orbital_entries.append((index, tokens[0]))

    pp_assignments = _assign_reference_paths(
        [reference for _, reference, _ in pp_entries],
        pp_orb_paths,
        orbital=False,
        labels=[label for _, _, label in pp_entries],
    )
    orbital_assignments = _assign_reference_paths(
        [reference for _, reference in orbital_entries],
        pp_orb_paths,
        orbital=True,
        labels=[label for _, _, label in pp_entries[: len(orbital_entries)]],
    )

    for (line_index, _, _), path in zip(pp_entries, pp_assignments):
        if path is not None:
            lines[line_index] = re.sub(
                r"^(\s*\S+\s+\S+\s+)\S+",
                lambda match: f"{match.group(1)}{path.name}",
                lines[line_index],
                count=1,
            )
    for (line_index, _), path in zip(orbital_entries, orbital_assignments):
        if path is not None:
            lines[line_index] = re.sub(
                r"^(\s*)\S+",
                lambda match: f"{match.group(1)}{path.name}",
                lines[line_index],
                count=1,
            )

    stru_path.write_text("".join(lines), encoding="utf-8")


def _prepare_run_directory(
    temp_path: Path,
    stru_path: Path,
    pp_orb_paths: List[Path],
    other_file_path: Optional[Path],
) -> None:
    run_stru_path = temp_path / "STRU"
    shutil.copy2(stru_path, run_stru_path)
    for path in pp_orb_paths:
        shutil.copy2(path, temp_path / path.name)
    _update_stru_file_references(run_stru_path, pp_orb_paths)
    if other_file_path:
        _copy_extra_path(other_file_path, temp_path / other_file_path.name)


def _execute_abacus(command: str, run_path: Path, stage: str) -> None:
    result = sp.run(
        shlex.split(command),
        cwd=run_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        full_output = "\n".join(
            ["===== STDOUT =====", result.stdout, "===== STDERR =====", result.stderr]
        )
        raise RuntimeError(
            f"ABACUS {stage} execution failed with exit code "
            f"{result.returncode}:\n{full_output}"
        )


def _collect_output(temp_path: Path, work_path: Path) -> dict:
    """Preserve the complete calculation root for downstream dftio parsing."""
    source = temp_path / "OUT.ABACUS"
    if not source.is_dir():
        raise FileNotFoundError(f"ABACUS did not create the expected output: {source}")
    shutil.copytree(temp_path, work_path, dirs_exist_ok=True)
    return {"abacus_output_path": work_path}


def _run_abacus(
    stru_path: Path,
    input_path: Path,
    kpt_path: Optional[Path] = None,
    command: str = "abacus",
    pp_orb_paths: Optional[List[Path]] = None,
    other_file_path: Optional[Path] = None,
):
    """Run one ABACUS calculation and return its complete working directory."""
    work_path = Path(generate_work_path()).absolute()
    pp_orb_paths = pp_orb_paths or []

    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)
        _prepare_run_directory(temp_path, stru_path, pp_orb_paths, other_file_path)
        shutil.copy2(input_path, temp_path / "INPUT")
        if kpt_path:
            shutil.copy2(kpt_path, temp_path / "KPT")

        _execute_abacus(command, temp_path, "calculation")
        return _collect_output(temp_path, work_path)


def _run_abacus_scf_nscf(
    stru_path: Path,
    scf_input_path: Path,
    nscf_input_path: Path,
    nscf_kpt_path: Path,
    command: str = "abacus",
    pp_orb_paths: Optional[List[Path]] = None,
    other_file_path: Optional[Path] = None,
):
    """Run SCF with kspacing, then NSCF with a K-line file in one directory.

    Keeping both stages in the same directory lets ``INPUT_nscf`` use
    ``read_file_dir ./OUT.ABACUS/`` and ``init_chg file`` to consume the charge
    density generated by SCF.
    """
    work_path = Path(generate_work_path()).absolute()
    pp_orb_paths = pp_orb_paths or []

    with tempfile.TemporaryDirectory(dir=work_path) as temp_dir:
        temp_path = Path(temp_dir)
        _prepare_run_directory(temp_path, stru_path, pp_orb_paths, other_file_path)

        shutil.copy2(scf_input_path, temp_path / "INPUT")
        kpt_in_run = temp_path / "KPT"
        if kpt_in_run.exists():
            kpt_in_run.unlink()
        _execute_abacus(command, temp_path, "SCF")

        if not (temp_path / "OUT.ABACUS").is_dir():
            raise FileNotFoundError("ABACUS SCF did not create OUT.ABACUS for NSCF")

        shutil.copy2(nscf_input_path, temp_path / "INPUT")
        shutil.copy2(nscf_kpt_path, kpt_in_run)
        _execute_abacus(command, temp_path, "NSCF")

        return _collect_output(temp_path, work_path)


def _resolve_abacus_output(abacus_out_path: Path) -> Path:
    """Accept either a calculation root or its OUT.ABACUS directory."""
    path = Path(abacus_out_path)
    output = path / "OUT.ABACUS"
    return output if output.is_dir() else path


def _abacus_get_fermi_level(abacus_out_path: Path):
    """Extract EFERMI, preferring the NSCF log."""
    abacus_out_path = _resolve_abacus_output(abacus_out_path)
    nscf_log = abacus_out_path / "running_nscf.log"
    if nscf_log.exists():
        log_file = nscf_log
    else:
        log_files = list(abacus_out_path.glob("running_*.log"))
        if not log_files:
            raise FileNotFoundError("No running_nscf.log or running_*.log was found")
        log_file = sorted(log_files, key=lambda path: ("scf" not in path.name, path.name))[0]

    pattern = re.compile(r"EFERMI\s*=\s*([-\d.Ee+]+)")
    with log_file.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                return {"fermi_level": float(match.group(1))}
    raise RuntimeError(f"EFERMI was not found in {log_file.name}")


def _load_abacus_bands(abacus_out_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ABACUS text bands as k-path coordinates and ``(Nk, Nband)`` energies."""
    band_file = _resolve_abacus_output(abacus_out_path) / "BANDS_1.dat"
    if not band_file.exists():
        raise FileNotFoundError(f"ABACUS band file not found: {band_file}")
    data = np.loadtxt(band_file, ndmin=2)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Invalid ABACUS band data shape {data.shape}; expected index, k-path, and bands"
        )
    return data[:, 1], data[:, 2:]


def _abacus_band_plot(abacus_out_path: Path, emin: float = None, emax: float = None):
    work_path = Path(generate_work_path()).absolute()
    fermi_level = _abacus_get_fermi_level(abacus_out_path)["fermi_level"]
    kpoints, bands = _load_abacus_bands(abacus_out_path)
    bands = bands - fermi_level

    plt.figure(figsize=(6, 5))
    for band_index in range(bands.shape[1]):
        plt.plot(kpoints, bands[:, band_index], color="black", linewidth=1)
    plt.axhline(0, linestyle="--")
    plt.ylim(bands.min() if emin is None else emin, bands.max() if emax is None else emax)
    plt.xlabel("k-path")
    plt.ylabel("Energy (eV)")
    plt.title("Band Structure")
    band_img_path = work_path / f"band_{int(time.time())}.png"
    plt.tight_layout()
    plt.savefig(band_img_path, dpi=300)
    plt.close()
    return {"abacus_output_path": band_img_path}


def _abacus_band_gap(
    abacus_out_path: Path,
    use_smart_band_gap: bool = False,
    smart_band_gap_atol: float = 1e-3,
    smart_band_gap_max_diff: float = 1.0,
):
    """Calculate a conventional or DeePTB smart gap from ABACUS text bands."""
    fermi_level = _abacus_get_fermi_level(abacus_out_path)["fermi_level"]
    _, bands = _load_abacus_bands(abacus_out_path)

    if use_smart_band_gap:
        # Local import avoids a module cycle: DeePTB band plotting also uses the
        # ABACUS Fermi-level parser for comparison plots.
        from dptb_pilot.tools.modules.deeptb.submodules.band import smart_band_gap

        smart_result = smart_band_gap(
            bands,
            fermi_level,
            atol=smart_band_gap_atol,
            max_diff=smart_band_gap_max_diff,
        )
        if smart_result is None:
            return {"band_gap": 0.0, "vbm": None, "cbm": None}
        return {
            **smart_result,
            "band_gap": float(smart_result["gap_width"]),
            "vbm": float(smart_result["gap_lower"] - fermi_level),
            "cbm": float(smart_result["gap_upper"] - fermi_level),
        }

    bands_relative = bands - fermi_level
    energies = bands_relative.ravel()
    conduction = energies[energies > 0]
    valence = energies[energies <= 0]
    if conduction.size == 0 or valence.size == 0:
        return {"band_gap": 0.0, "vbm": None, "cbm": None}
    cbm = float(np.min(conduction))
    vbm = float(np.max(valence))
    return {"band_gap": cbm - vbm, "vbm": vbm, "cbm": cbm}
