from pathlib import Path

from dptb_pilot.tools.modules.dpnegf.submodules.archive import pack_files, unpack_files


def test_dpnegf_archive_roundtrip(tmp_path: Path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (work_dir / "b.txt").write_text("beta", encoding="utf-8")

    archive = pack_files(work_dir, ["a.txt", "b.txt"], "files.tar.gz")
    assert archive.exists()

    out_dir = tmp_path / "out"
    extracted = unpack_files(archive, out_dir)
    assert sorted(path.name for path in extracted) == ["a.txt", "b.txt"]
    assert (out_dir / "a.txt").read_text(encoding="utf-8") == "alpha"
    assert (out_dir / "b.txt").read_text(encoding="utf-8") == "beta"
