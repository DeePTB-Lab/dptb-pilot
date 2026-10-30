from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules import dftio


def _fake_result(command, cwd, capture_output, text):
    output = Path(command[command.index("-o") + 1])
    output.mkdir(parents=True)
    (output / "Si.0").mkdir()
    (output / "Si.1").mkdir()
    return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def test_dftio_parse_normalizes_single_parent_and_maps_inputs(monkeypatch, tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    first = root / "calculation one"
    second = root / "calculation two"
    first.mkdir()
    second.mkdir()
    (first / "INPUT").write_text("input", encoding="utf-8")
    (second / "INPUT").write_text("input", encoding="utf-8")

    work_path = tmp_path / "work"
    work_path.mkdir()
    monkeypatch.setattr(dftio, "generate_work_path", lambda: str(work_path))
    monkeypatch.setattr(dftio.sp, "run", _fake_result)

    result = dftio._dftio_parse(root, prefix="does-not-match")

    assert result["output_path"] == work_path / "parse_result"
    assert Path(result["input_mapping"][str(first.resolve())]).name == "Si.0"
    assert Path(result["input_mapping"][str(second.resolve())]).name == "Si.1"


def test_dftio_parse_accepts_multiple_explicit_directories(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    work_path = tmp_path / "work"
    work_path.mkdir()
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return _fake_result(command, **kwargs)

    monkeypatch.setattr(dftio, "generate_work_path", lambda: str(work_path))
    monkeypatch.setattr(dftio.sp, "run", fake_run)

    result = dftio._dftio_parse([first, second], output_dir_name="parsed")

    command = commands[0]
    assert command[command.index("-p") + 1] == "dftio_input"
    assert result["output_path"] == work_path / "parsed"
    assert len(result["input_mapping"]) == 2


def test_dftio_path_annotation_supports_single_or_multiple_paths():
    from dptb_pilot.tools.artifacts import _convert_paths, _path_annotation
    from dptb_pilot.tools.modules.deeptb.dftio import dftio_parse

    annotation = _path_annotation(dftio_parse.__annotations__["work_root"])
    assert annotation[0].__name__ in {"UnionType", "Union"}

    class Storage:
        def download(self, key, destination):
            target = Path(destination) / Path(key).name
            target.mkdir(parents=True, exist_ok=True)
            return str(target)

    single = _convert_paths("one", annotation, Path("inputs"), "local", Storage())
    multiple = _convert_paths(["one", "two"], annotation, Path("inputs"), "local", Storage())
    assert isinstance(single, Path)
    assert isinstance(multiple, list)
    assert all(isinstance(item, Path) for item in multiple)
