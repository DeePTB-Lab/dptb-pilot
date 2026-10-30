import shutil
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write, read
import subprocess as sp

from dptb_pilot.tools.modules.util.comm import generate_work_path


def write_lammps_data(ase_atoms: Atoms, path: str, specorder=None):
    """
    将 ASE Atoms 写为 lammps.data（lammps-data 格式）。
    specorder: 可选的元素顺序列表以控制 type 映射
    """
    if specorder is None:
        specorder = []
        for s in ase_atoms.get_chemical_symbols():
            if s not in specorder:
                specorder.append(s)
    write(path, ase_atoms, format='lammps-data', specorder=specorder)
    return specorder

def generate_group_lines_by_ranges(mobile_count, fixed_ids=[], indenter_ids=[]):
    """
    生成较为简单的 group 定义文本，fixed_ids, indenter_ids 是 atom id (1-based) 列表
    """
    lines = []
    if fixed_ids:
        ids = " ".join(map(str, sorted(set(fixed_ids))))
        lines.append(f"group fixed id {ids}")
        lines.append("group mobile subtract all fixed")
    else:
        lines.append("group mobile all")
    if indenter_ids:
        ids = " ".join(map(str, sorted(set(indenter_ids))))
        lines.append(f"group indenter id {ids}")
        # remove indenter from mobile
        lines.append("group mobile subtract indenter")
    return "\n".join(lines)

def _run_lammps(in_lammps_file_path:Path,
                lammps_data_file_path:Path,
                deepmd_model_file_path:Path=None,
                lmp_command:str='lmp'):
    """
    Run a LAMMPS calculation in a temporary directory and collect the relaxed structure.

    Parameters
    ----------
    in_lammps_file_path : Path
        Path to the LAMMPS input script. The script is copied into the execution
        directory and run as ``in.lammps``.
    lammps_data_file_path : Path
        Path to the LAMMPS data file referenced by the input script.
    deepmd_model_file_path : Path, optional
        DeepMD model file to copy next to the LAMMPS input files.
    lmp_command : str, optional
        LAMMPS executable command, by default ``lmp``.

    Returns
    -------
    dict
        Dictionary with ``relaxed_system_file_path`` pointing to the saved VASP file.

    Raises
    ------
    RuntimeError
        If the LAMMPS command exits with a non-zero status.
    """
    work_path = Path(generate_work_path()).absolute()

    # Use a temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        shutil.copy(in_lammps_file_path, temp_path / in_lammps_file_path.name)
        shutil.copy(lammps_data_file_path, temp_path / lammps_data_file_path.name)
        if deepmd_model_file_path:
            shutil.copy(deepmd_model_file_path, temp_path / deepmd_model_file_path.name)

        # Run dptb command in temp dir
        cmd = [lmp_command, "-i", 'in.lammps', "-log", "log.lammps"]
        result = sp.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"lammps execution failed:\n{result.stderr}")

        relaxed_system = read("relaxed.data", format='lammps-data')
        write("relaxed.vasp", relaxed_system, vasp5=True)

        # Copy result back to work_path
        import time
        timestamp = int(time.time())
        relaxed_system_filename = f"relaxed_{timestamp}.vasp"
        output_relaxed_system_path = work_path / relaxed_system_filename
        shutil.copy("relaxed.vasp", output_relaxed_system_path)

    return {"relaxed_system_file_path": Path(output_relaxed_system_path)}