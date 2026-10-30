from pathlib import Path
from typing import TypedDict
from dptb_pilot.tools.init import mcp

class FilePathResult(TypedDict):
    file_path: Path

@mcp.tool()
def get_file_path(
        file_name: str = "your_file_name",
        work_path: str = "."
) -> FilePathResult:
    """
    根据本地文件名和工作路径获得文件路径。（由于运行的work_path是程序生成的，则需要调用该函数进行得到实际的Path）

    参数:
        file_name: 文件名
        work_path: 工作路径，为文件夹（在本程序中由程序自动输入，而不是大语言模型输入）

    返回:
        本地文件的Path。

    抛出:
        AssumptionError: 文件名为空。
        FileNotFoundError: 文件未找到。
    """

    assert file_name != "your_file_name", "文件名为空"

    # Resolve absolute paths
    work_dir = Path(work_path).absolute()
    file_path = work_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return {"file_path": Path(file_path).absolute()}
