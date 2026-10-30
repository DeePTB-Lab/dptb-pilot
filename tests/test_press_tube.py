from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.press_tube import build_and_generate


def test_build_and_generate_press_tube_task(tmp_path):
    structure = Path(__file__).parent / "resources" / "10_0.vasp"
    output = tmp_path / "tasks"

    build_and_generate(
        poscar_path=str(structure),
        out_root=str(output),
        n_repeat=1,
        indent_depths=[6.0],
        nsteps=1,
    )

    task_directories = [path for path in output.iterdir() if path.is_dir()]
    assert len(task_directories) == 1
    assert (task_directories[0] / "lammps.data").is_file()
    assert (task_directories[0] / "in.lammps").is_file()
