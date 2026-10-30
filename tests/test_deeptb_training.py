import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dptb_pilot.tools.modules.deeptb.submodules import training


def _write_config(tmp_path: Path, data_options: dict) -> Path:
    config_path = tmp_path / "input.json"
    config_path.write_text(
        json.dumps(
            {
                "common_options": {"basis": {"Si": "1s1p"}},
                "model_options": {},
                "train_options": {"num_epoch": 1},
                "data_options": data_options,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _dataset_root(tmp_path: Path, folder_name: str) -> tuple[Path, Path]:
    root = tmp_path / "dftio_data"
    folder = root / folder_name
    folder.mkdir(parents=True)
    (root / "info.json").write_text(
        json.dumps({"nframes": 1, "natoms": 1, "pbc": True, "pos_type": "cart"}),
        encoding="utf-8",
    )
    (folder / "positions.dat").write_text("0 0 0\n", encoding="utf-8")
    (folder / "atomic_numbers.dat").write_text("14\n", encoding="utf-8")
    return root, folder


def _fake_successful_run(calls):
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_path = Path(kwargs["cwd"]) / command[command.index("-o") + 1]
        checkpoint = output_path / "checkpoint" / "model.best.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("trained", encoding="utf-8")
        return CompletedProcess(command, 0, stdout="trained", stderr="")

    return fake_run


def test_e3_training_rewrites_remote_dataset_root_and_checks_matrix_data(
    monkeypatch, tmp_path
):
    dataset_root, folder = _dataset_root(tmp_path, "Si64.0")
    (folder / "hamiltonians.h5").write_bytes(b"ham")
    (folder / "overlaps.h5").write_bytes(b"overlap")
    config_path = _write_config(
        tmp_path,
        {
            "train": {
                "root": "/local/path/that/does/not/exist",
                "prefix": "Si64",
                "get_Hamiltonian": True,
                "get_overlap": True,
            }
        },
    )
    calls = []
    monkeypatch.setattr(training.subprocess, "run", _fake_successful_run(calls))

    result = training.train_with_cli(
        config_path,
        dataset_root,
        work_path=str(tmp_path / "remote_job"),
    )

    rewritten = json.loads(result["config_path"].read_text(encoding="utf-8"))
    run_path = result["config_path"].parent
    remote_root = Path(rewritten["data_options"]["train"]["root"])
    assert remote_root == (run_path / "datasets" / "train").resolve()
    assert remote_root != Path("/local/path/that/does/not/exist")
    assert (remote_root / "Si64.0" / "hamiltonians.h5").is_file()
    assert (remote_root / "Si64.0" / "overlaps.h5").is_file()
    assert calls[0][0] == ["dptb", "train", "input.json", "-o", "output"]
    assert calls[0][1]["cwd"] == run_path
    assert result["model_path"].name == "model.best.pth"


def test_sk_training_uses_same_tool_and_validates_eigenvalue_data(monkeypatch, tmp_path):
    dataset_root, folder = _dataset_root(tmp_path, "Si.0")
    (folder / "eigenvalues.npy").write_bytes(b"eigenvalues")
    (folder / "kpoints.npy").write_bytes(b"kpoints")
    config_path = _write_config(
        tmp_path,
        {
            "train": {
                "root": "../old-data",
                "prefix": "Si",
                "type": "DefaultDataset",
                "get_eigenvalues": True,
                "get_Hamiltonian": False,
            }
        },
    )
    calls = []
    monkeypatch.setattr(training.subprocess, "run", _fake_successful_run(calls))

    result = training.train_with_cli(
        config_path,
        dataset_root,
        work_path=str(tmp_path / "remote_job"),
    )

    rewritten = json.loads(result["config_path"].read_text(encoding="utf-8"))
    staged_root = Path(rewritten["data_options"]["train"]["root"])
    assert (staged_root / "Si.0" / "eigenvalues.npy").is_file()
    assert (staged_root / "Si.0" / "kpoints.npy").is_file()
    assert "trained" in result["log_path"].read_text(encoding="utf-8")


def test_validation_section_reuses_transferred_root_when_not_supplied(
    monkeypatch, tmp_path
):
    dataset_root, train_folder = _dataset_root(tmp_path, "train.0")
    validation_folder = dataset_root / "validation.0"
    validation_folder.mkdir()
    for folder in (train_folder, validation_folder):
        (folder / "hamiltonians.h5").write_bytes(b"ham")
        if folder == validation_folder:
            (folder / "positions.dat").write_text("0 0 0\n", encoding="utf-8")
            (folder / "atomic_numbers.dat").write_text("14\n", encoding="utf-8")
    config_path = _write_config(
        tmp_path,
        {
            "train": {
                "root": "/old/train",
                "prefix": "train",
                "get_Hamiltonian": True,
            },
            "validation": {
                "root": "/old/validation",
                "prefix": "validation",
                "get_Hamiltonian": True,
            },
        },
    )
    monkeypatch.setattr(training.subprocess, "run", _fake_successful_run([]))

    result = training.train_with_cli(
        config_path,
        dataset_root,
        work_path=str(tmp_path / "remote_job"),
    )

    rewritten = json.loads(result["config_path"].read_text(encoding="utf-8"))
    assert rewritten["data_options"]["validation"]["root"] == rewritten[
        "data_options"
    ]["train"]["root"]


def test_training_rejects_dataset_missing_requested_type(monkeypatch, tmp_path):
    dataset_root, folder = _dataset_root(tmp_path, "Si64.0")
    (folder / "hamiltonians.h5").write_bytes(b"ham")
    config_path = _write_config(
        tmp_path,
        {
            "train": {
                "root": "/old/data",
                "prefix": "Si64",
                "get_Hamiltonian": True,
                "get_overlap": True,
            }
        },
    )
    monkeypatch.setattr(
        training.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("CLI should not run"),
    )

    with pytest.raises(ValueError, match="get_overlap.*overlaps.h5"):
        training.train_with_cli(
            config_path,
            dataset_root,
            work_path=str(tmp_path / "remote_job"),
        )
