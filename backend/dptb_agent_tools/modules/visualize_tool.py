import os
import json
from dptb_agent_tools.init_mcp import mcp

@mcp.tool()
def visualize_structure(file_name: str, work_path: str) -> str:
    """
    Visualize a crystal structure file (POSCAR, CIF, etc.) in the chat interface.
    
    Args:
        file_name: The name of the file to visualize (e.g., POSCAR, structure.cif).
        work_path: The absolute path to the workspace directory.
        
    Returns:
        A special markdown block that the frontend uses to render the structure.
    """
    file_path = os.path.join(work_path, file_name)
    
    if not os.path.exists(file_path):
        return f"Error: File {file_name} not found in {work_path}"
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Determine format
        ext = os.path.splitext(file_name)[1].lower()
        format_type = 'vasp' # default
        if 'cif' in ext:
            format_type = 'cif'
        elif 'xyz' in ext:
            format_type = 'xyz'
        elif 'poscar' in file_name.lower() or 'vasp' in ext:
            format_type = 'vasp'
            
        # Construct the visualization block
        payload = {
            "format": format_type,
            "data": content
        }
        
        return f":::visualize\n{json.dumps(payload)}\n:::"
        
    except Exception as e:
        return f"Error reading file for visualization: {str(e)}"
