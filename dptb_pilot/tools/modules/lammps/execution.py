import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional

from ase.io import read, write


def execute_lammps(
        in_lammps_file_path: Path,
        lammps_data_file_path: Path,
        deepmd_model_file_path: Optional[Path] = None,
        lmp_command: str = "lmp",
        output_path: Path = Path("."),
        task_name: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
) -> dict:
    input_file = Path(in_lammps_file_path).expanduser().resolve()
    data_file = Path(lammps_data_file_path).expanduser().resolve()
    model_file = Path(deepmd_model_file_path).expanduser().resolve() if deepmd_model_file_path else None
    for path, label in ((input_file, "LAMMPS input"), (data_file, "LAMMPS data")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if model_file is not None and not model_file.is_file():
        raise FileNotFoundError(f"DeepMD model file not found: {model_file}")

    output_root = Path(output_path).expanduser().resolve()
    task_name = task_name or f"lammps_run_{uuid.uuid4().hex[:8]}"
    if Path(task_name).name != task_name or task_name in {".", ".."}:
        raise ValueError("task_name must be a single directory name")
    work_dir = output_root / task_name
    work_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(input_file, work_dir / "in.lammps")
    shutil.copy2(data_file, work_dir / "lammps.data")
    if model_file is not None:
        shutil.copy2(model_file, work_dir / model_file.name)

    command = shlex.split(lmp_command, posix=os.name != "nt")
    if not command:
        raise ValueError("lmp_command cannot be empty")
    command.extend(["-i", "in.lammps", "-log", "log.lammps"])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    result = subprocess.run(command, cwd=work_dir, capture_output=True, text=True, check=False)
    (work_dir / "stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (work_dir / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    log_path = work_dir / "log.lammps"
    if result.returncode != 0:
        raise RuntimeError(
            f"LAMMPS exited with status {result.returncode}; command: {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if not log_path.is_file():
        raise FileNotFoundError(f"LAMMPS completed without creating a log: {log_path}")

    relaxed_data = work_dir / "relaxed.data"
    relaxed_structure = None
    if relaxed_data.is_file():
        atoms = read(relaxed_data, format="lammps-data")
        relaxed_structure = work_dir / "relaxed.vasp"
        write(relaxed_structure, atoms, format="vasp", vasp5=True)
    return {
        "task_path": work_dir,
        "log_file_path": log_path,
        "relaxed_data_file_path": relaxed_data if relaxed_data.is_file() else None,
        "relaxed_structure_file_path": relaxed_structure,
        "command": shlex.join(command),
    }
