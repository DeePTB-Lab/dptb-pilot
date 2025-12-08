import os
from dptb_pilot.tools.init import mcp

@mcp.tool()
def list_workspace_files(work_path: str) -> str:
    """
    List all files in the current workspace directory.
    Useful for checking uploaded files (e.g. POSCARs) or generated results.
    
    Args:
        work_path: The absolute path to the workspace directory.
        
    Returns:
        A formatted string listing the files and their sizes.
    """
    if not os.path.isabs(work_path):
        return f"Error: work_path must be an absolute path. Got: {work_path}"
        
    if not os.path.exists(work_path):
        return f"Error: Directory {work_path} does not exist."
        
    if not os.path.isdir(work_path):
        return f"Error: {work_path} is not a directory."
    
    try:
        files = os.listdir(work_path)
        files.sort()
        
        if not files:
            return "Workspace is empty."
            
        result = f"Files in {work_path}:\n"
        for item in files:
            item_path = os.path.join(work_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                result += f"[FILE] {item} ({size} bytes)\n"
            elif os.path.isdir(item_path):
                result += f"[DIR]  {item}\n"
                
        return result
    except Exception as e:
        return f"Error listing workspace: {str(e)}"

@mcp.tool()
def read_file_content(file_path: str) -> str:
    """
    Read the content of a file.
    
    Args:
        file_path: The absolute path to the file.
        
    Returns:
        The content of the file as a string.
    """
    if not os.path.isabs(file_path):
        return f"Error: file_path must be an absolute path. Got: {file_path}"
        
    if not os.path.exists(file_path):
        return f"Error: File {file_path} does not exist."
        
    if not os.path.isfile(file_path):
        return f"Error: {file_path} is not a file."
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
