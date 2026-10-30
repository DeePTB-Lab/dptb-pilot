import shutil
from pathlib import Path
from typing import TypedDict

from dptb_pilot.tools.init import mcp


class FilePathResult(TypedDict):
    file_path: str
    file_name: str


@mcp.tool()
def download_artifact(
    artifact_url: Path,
    work_path: str = ".",
) -> FilePathResult:
    """Download a Bohrium artifact into a local workspace directory.

    ``artifact_url`` accepts a ``bohrium://`` URI. CalculationMCPServer first
    resolves it with the storage credentials configured for the current
    session, then passes the downloaded local path to this function.

    Pass the session's absolute ``files`` directory as ``work_path`` when the
    downloaded file needs to be exposed through the Pilot download API.

    Parameters
    ----------
    artifact_url : Path
        Resolved local path supplied from a ``bohrium://`` artifact URI by the
        MCP server.
    work_path : str, optional
        Destination workspace directory. Existing directories are merged and
        existing files with the same name are replaced.

    Returns
    -------
    FilePathResult
        Absolute staged path and its basename.
    """
    source = Path(artifact_url).resolve()
    if not source.exists():
        raise ValueError(f"Downloaded artifact does not exist: {source}")

    work_dir = Path(work_path).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    destination = work_dir / source.name

    try:
        if source != destination:
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
        return {
            "file_path": str(destination),
            "file_name": destination.name,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to stage downloaded artifact: {exc}") from exc
