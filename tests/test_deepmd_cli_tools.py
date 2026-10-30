from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dptb_pilot.tools.modules.deepmd import cli


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "data" / "set.000").mkdir(parents=True)
    (workspace / "input.json").write_text(
        '{"training": {"training_data": {"systems": ["data"]}}}',
        encoding="utf-8",
    )
    return workspace


def test_train_deepmd_model_uses_native_cli(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        (Path(kwargs["cwd"]) / "model.ckpt.pt").write_text("checkpoint")
        return CompletedProcess(command, 0, stdout="trained", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.train_with_cli(
        workspace,
        work_path=str(tmp_path / "outputs"),
        skip_neighbor_stat=True,
        extra_cli_args=["--log-level", "INFO"],
    )

    assert calls[0][0] == [
        "dp",
        "--pt",
        "train",
        "input.json",
        "--skip-neighbor-stat",
        "--log-level",
        "INFO",
    ]
    assert calls[0][1]["cwd"] == result["output_path"]
    assert (result["output_path"] / "data" / "set.000").is_dir()
    assert result["model_path"].name == "model.ckpt.pt"
    assert "trained" in result["log_path"].read_text(encoding="utf-8")


def test_finetune_dpa3_downloads_builtin_model_and_runs_pt_cli(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        cwd = Path(kwargs["cwd"])
        if command[1:3] == ["pretrained", "download"]:
            cache_dir = Path(command[command.index("--cache-dir") + 1])
            cache_dir.mkdir(parents=True)
            (cache_dir / "DPA-3.3-1M.pt").write_text("pretrained")
            return CompletedProcess(command, 0, stdout=str(cache_dir), stderr="")
        (cwd / "model.ckpt.pt").write_text("finetuned")
        return CompletedProcess(command, 0, stdout="finetuned", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli.finetune_dpa3_with_cli(
        workspace,
        work_path=str(tmp_path / "outputs"),
        model_branch="Materials",
    )

    assert calls[0][0:4] == ["dp", "pretrained", "download", "DPA-3.3-1M"]
    assert calls[1] == [
        "dp",
        "--pt",
        "train",
        "input.json",
        "--finetune",
        str(Path("pretrained") / "DPA-3.3-1M.pt"),
        "--use-pretrain-script",
        "--model-branch",
        "Materials",
    ]
    assert result["pretrained_model_path"].name == "DPA-3.3-1M.pt"
    assert result["model_path"].read_text() == "finetuned"


def test_training_config_cannot_escape_workspace(monkeypatch, tmp_path):
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("CLI should not run"),
    )

    with pytest.raises(ValueError, match="inside the training workspace"):
        cli.train_with_cli(
            workspace,
            input_config="../input.json",
            work_path=str(tmp_path / "outputs"),
        )
