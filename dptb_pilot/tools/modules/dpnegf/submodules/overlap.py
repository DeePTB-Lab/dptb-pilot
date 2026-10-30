import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from dptb_pilot.tools.modules.util.comm import run_command, temporary_chdir


def get_abacus_overlap(
        poscar_file_path: Path,
        input_file_path: Path,
        pp_file_paths: List[Path],
        orb_file_paths: List[Path],
        run_config: Dict[str, Any],
        work_path: str = "."
) -> Dict[str, Any]:
    """Run ABACUS get_S for one POSCAR and collect overlap inputs/outputs."""
    import dpdata

    poscar_file_path = Path(poscar_file_path).absolute()
    input_file_path = Path(input_file_path).absolute()
    pp_file_paths = [Path(path).absolute() for path in pp_file_paths]
    orb_file_paths = [Path(path).absolute() for path in orb_file_paths]
    work_dir = Path(work_path).absolute() / f"getS_{time.time()}"
    work_dir.mkdir(parents=True, exist_ok=True)

    with temporary_chdir(work_dir):
        dpdata.System(poscar_file_path, fmt="vasp/poscar").to(
            "abacus/stru",
            "STRU",
            pp_file=pp_file_paths,
            numerical_orbital=orb_file_paths,
        )
        shutil.copy(input_file_path, work_dir / "INPUT")
        ret, out, err = run_command(run_config.get("command", "abacus"), shell=True)
        if ret != 0:
            raise RuntimeError(f"abacus failed\ncommand was: {run_config.get('command', 'abacus')}\nout msg: {out}\nerr msg: {err}")

    out_dir = work_dir / "OUT.ABACUS"
    return {
        "stru_path": (out_dir / "STRU.cif").absolute(),
        "input_path": (out_dir / "INPUT").absolute(),
        "running_log_path": (out_dir / "running_nscf.log").absolute(),
        "overlap_csr_path": (out_dir / "data-SR-sparse_SPIN0.csr").absolute(),
    }


def convert_overlap(
        stru_path: Path,
        input_path: Path,
        running_log_path: Path,
        overlap_csr_path: Path,
        work_path: str = "."
) -> Dict[str, Any]:
    """Convert ABACUS sparse overlap CSR output to dftio overlaps.h5."""
    from dftio.io.parse import ParserRegister

    stru_path = Path(stru_path).absolute()
    input_path = Path(input_path).absolute()
    running_log_path = Path(running_log_path).absolute()
    overlap_csr_path = Path(overlap_csr_path).absolute()
    work_dir = Path(work_path).absolute() / f"convert_overlap_{time.time()}"
    out_abacus_dir = work_dir / "OUT.ABACUS"
    out_abacus_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(stru_path, out_abacus_dir / "STRU.cif")
    shutil.copy(input_path, out_abacus_dir / "INPUT")
    shutil.copy(running_log_path, out_abacus_dir / "running_nscf.log")
    shutil.copy(overlap_csr_path, out_abacus_dir / "data-SR-sparse_SPIN0.csr")

    with temporary_chdir(work_dir):
        args = {
            "command": "parse",
            "log_level": 20,
            "log_path": None,
            "mode": "abacus",
            "num_workers": 1,
            "root": "../",
            "prefix": work_dir.name,
            "outroot": "convert_result",
            "format": "dat",
            "hamiltonian": False,
            "overlap": True,
            "density_matrix": False,
            "eigenvalue": False,
            "band_index_min": 0,
            "energy": False,
        }
        parser = ParserRegister(**args)
        for i in range(len(parser)):
            parser.write(idx=i, **args)

    convert_root = work_dir / "convert_result"
    subdirs = [path for path in convert_root.iterdir() if path.is_dir()]
    if not subdirs:
        raise RuntimeError(f"dftio conversion did not produce a result directory under {convert_root}")
    overlap_h5_path = subdirs[0] / "overlaps.h5"
    if not overlap_h5_path.exists():
        raise RuntimeError(f"dftio conversion did not produce overlaps.h5 at {overlap_h5_path}")
    return {"overlap_h5_path": overlap_h5_path.absolute()}
