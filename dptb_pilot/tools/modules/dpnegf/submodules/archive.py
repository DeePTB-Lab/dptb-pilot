import tarfile
from pathlib import Path
from typing import Iterable, List, Optional, Union


def pack_files(
        work_dir: Union[str, Path],
        file_names: Iterable[Union[str, Path]],
        archive_name: str,
        output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Pack existing files under ``work_dir`` into a gzipped tar archive."""
    work_dir = Path(work_dir).absolute()
    output_dir = Path(output_dir).absolute() if output_dir is not None else work_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name

    with tarfile.open(archive_path, "w:gz") as tar:
        for file_name in file_names:
            file_path = Path(file_name)
            if not file_path.is_absolute():
                file_path = work_dir / file_path
            if not file_path.exists():
                continue
            tar.add(file_path, arcname=file_path.name)

    return archive_path.absolute()


def unpack_files(archive_file_path: Union[str, Path], unpack_dir: Union[str, Path]) -> List[Path]:
    """Unpack a tar/tar.gz archive and return extracted file paths."""
    archive_file_path = Path(archive_file_path).absolute()
    unpack_dir = Path(unpack_dir).absolute()
    unpack_dir.mkdir(parents=True, exist_ok=True)

    before = {path.resolve() for path in unpack_dir.rglob("*")}
    with tarfile.open(archive_file_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (unpack_dir / member.name).resolve()
            if not str(target).startswith(str(unpack_dir.resolve())):
                raise RuntimeError(f"Refusing to unpack unsafe archive member: {member.name}")
        tar.extractall(unpack_dir)

    extracted = []
    for path in unpack_dir.rglob("*"):
        if path.is_file() and path.resolve() not in before:
            extracted.append(path.absolute())
    return extracted
