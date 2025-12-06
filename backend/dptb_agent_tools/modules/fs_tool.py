import os
import pypdf
from typing import List, Optional
from dptb_agent_tools.init_mcp import mcp

@mcp.tool()
def list_directory(path: str) -> str:
    """
    List the contents of a directory.
    
    Args:
        path: The absolute or relative path to the directory.
        
    Returns:
        A string listing the files and directories, or an error message.
    """
    try:
        # Resolve path relative to current working directory if it's relative
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."
            
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory."
            
        items = os.listdir(path)
        items.sort()
        
        result = f"Contents of {path}:\n"
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result += f"[DIR]  {item}\n"
            else:
                result += f"[FILE] {item}\n"
                
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@mcp.tool()
def read_file_content(path: str) -> str:
    """
    Read the content of a file. Supports text files and PDFs.
    
    Args:
        path: The absolute or relative path to the file.
        
    Returns:
        The content of the file, or an error message.
    """
    try:
        # Resolve path relative to current working directory if it's relative
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
            
        if os.path.isdir(path):
            return f"Error: '{path}' is a directory. Use list_directory instead."
            
        if path.endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(path)
                content = f"--- PDF Content: {os.path.basename(path)} ---\n"
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        content += f"\n--- Page {i+1} ---\n{text}"
                return content
            except Exception as e:
                return f"Error reading PDF file: {str(e)}"
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                return "Error: File content is not valid UTF-8 text."
            except Exception as e:
                return f"Error reading text file: {str(e)}"
                
    except Exception as e:
        return f"Error accessing file: {str(e)}"

@mcp.tool()
def grep_files(path: str, pattern: str, case_sensitive: bool = False) -> str:
    """
    Search for a string pattern in files within a directory.
    
    Args:
        path: The directory to search in.
        pattern: The string pattern to search for.
        case_sensitive: Whether the search should be case-sensitive (default: False).
        
    Returns:
        A string listing the matching files and lines.
    """
    try:
        # Resolve path relative to current working directory if it's relative
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        if not os.path.exists(path):
            return f"Error: Path '{path}' does not exist."
            
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory."
            
        results = []
        count = 0
        max_results = 100  # Limit results to prevent context overflow
        
        for root, _, filenames in os.walk(path):
            # Skip hidden directories and tests/pycache
            if "/." in root or "/__pycache__" in root:
                continue
                
            for filename in filenames:
                # Only search text-based files
                if not (filename.endswith(".py") or filename.endswith(".md") or filename.endswith(".txt") or filename.endswith(".json") or filename.endswith(".toml")):
                    continue
                    
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines):
                        if count >= max_results:
                            results.append("... (Too many matches, search truncated)")
                            break
                            
                        match = False
                        if case_sensitive:
                            if pattern in line:
                                match = True
                        else:
                            if pattern.lower() in line.lower():
                                match = True
                                
                        if match:
                            rel_path = os.path.relpath(file_path, path)
                            results.append(f"{rel_path}:{i+1}: {line.strip()}")
                            count += 1
                            
                    if count >= max_results:
                        break
                except Exception:
                    # Ignore binary files or read errors
                    continue
            
            if count >= max_results:
                break
                
        if not results:
            return f"No matches found for pattern '{pattern}' in {path}."
            
        return "\n".join(results)
        
    except Exception as e:
        return f"Error executing grep: {str(e)}"
