import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import subprocess
from pathlib import Path
import uuid
import shutil
import pypdf
import ast
import json

# Configuration
# Assuming script is run from project root or installed as package
import dptb_pilot.tools
TOOLS_DIR = os.path.dirname(dptb_pilot.tools.__file__)
KNOWLEDGE_BASE_DIR = os.path.join(TOOLS_DIR, "data", "deeptb_knowledge")
REPO_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "repo")
PAPER_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "paper")
NOTEBOOK_PATH = os.path.join(KNOWLEDGE_BASE_DIR, "notebook")
CHROMA_DB_PATH = os.path.join(TOOLS_DIR, "data", "chroma_db")
COLLECTION_NAME = "deeptb_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def ensure_git_repo(
    base_dir: str,
    repo_url: str,
    folder_name: str | None = None
):
    """
    如果 base_dir/folder_name 不存在，则 git clone repo_url
    如果存在，则跳过

    Parameters
    ----------
    base_dir : str
        父目录路径
    repo_url : str
        git 仓库地址
    folder_name : str | None
        clone 后的文件夹名，None 表示使用仓库默认名
    """
    base_dir = Path(base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if folder_name is None:
        folder_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

    repo_path = base_dir / folder_name

    if repo_path.exists():
        print(f"[SKIP] Repository already exists: {repo_path}")
        return

    print(f"[CLONE] Cloning {repo_url} into {repo_path}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            check=True
        )
        print("[DONE] Clone successful")
    except subprocess.CalledProcessError as e:
        print("[ERROR] git clone failed")
        raise e


def process_python_file(file_path):
    """Parse Python file using AST to extract classes and functions."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        tree = ast.parse(content)
        filename = os.path.basename(file_path)
        
        # 1. Extract Module Docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            chunks.append({
                "text": f"File: {filename}\nType: Module Docstring\n\n{docstring}",
                "metadata": {"source": file_path, "filename": filename, "type": "code_doc"}
            })

        # 2. Extract Classes and Functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get full source of the function
                start_line = node.lineno
                end_line = node.end_lineno
                func_content = content.splitlines()[start_line-1:end_line]
                func_text = "\n".join(func_content)
                
                chunks.append({
                    "text": f"File: {filename}\nType: Function\nName: {node.name}\n\n{func_text}",
                    "metadata": {"source": file_path, "filename": filename, "type": "function", "name": node.name}
                })
                
            elif isinstance(node, ast.ClassDef):
                # Get full source of the class
                start_line = node.lineno
                end_line = node.end_lineno
                class_content = content.splitlines()[start_line-1:end_line]
                class_text = "\n".join(class_content)
                
                chunks.append({
                    "text": f"File: {filename}\nType: Class\nName: {node.name}\n\n{class_text}",
                    "metadata": {"source": file_path, "filename": filename, "type": "class", "name": node.name}
                })
                
    except Exception as e:
        print(f"Error parsing Python file {file_path}: {e}")
        # Fallback to simple chunking if AST fails
        return process_text_file(file_path, "code")
        
    return chunks

def process_notebook_file(file_path):
    """Parse Jupyter Notebook to extract code and markdown cells."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
            
        filename = os.path.basename(file_path)
        
        for i, cell in enumerate(notebook.get("cells", [])):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", []))
            
            if not source.strip():
                continue
                
            if cell_type == "markdown":
                chunks.append({
                    "text": f"File: {filename}\nType: Notebook Markdown\nCell: {i+1}\n\n{source}",
                    "metadata": {"source": file_path, "filename": filename, "type": "notebook_md", "page": i+1}
                })
            elif cell_type == "code":
                chunks.append({
                    "text": f"File: {filename}\nType: Notebook Code\nCell: {i+1}\n\n{source}",
                    "metadata": {"source": file_path, "filename": filename, "type": "notebook_code", "page": i+1}
                })
                
    except Exception as e:
        print(f"Error parsing Notebook {file_path}: {e}")
        
    return chunks

def process_text_file(file_path, type_label="doc"):
    """Simple chunking for text/markdown files."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        filename = os.path.basename(file_path)
        chunk_size = 1000
        overlap = 200
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i : i + chunk_size]
            if len(chunk) < 50:
                continue
                
            chunks.append({
                "text": f"File: {filename}\nType: {type_label}\n\n{chunk}",
                "metadata": {"source": file_path, "filename": filename, "type": type_label}
            })
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
    return chunks

def build_knowledge_base():
    print("Initializing embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Initializing ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    documents = []
    metadatas = []
    ids = []

    print("Scanning notebook for files...")
    for root, _, files in os.walk(NOTEBOOK_PATH):
        if "/." in root or "/tests" in root or "/__pycache__" in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_chunks = []

            if file.endswith(".py"):
                print(f"  [AST] Parsing Python: {file}")
                file_chunks = process_python_file(file_path)
            elif file.endswith(".ipynb"):
                print(f"  [NB]  Parsing Notebook: {file}")
                file_chunks = process_notebook_file(file_path)
            elif file.endswith(".md") or file.endswith(".txt") or file.endswith(".rst"):
                print(f"  [DOC] Reading Document: {file}")
                file_chunks = process_text_file(file_path, "doc")

            for chunk in file_chunks:
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                ids.append(str(uuid.uuid4()))

    print("Scanning repository for files...")
    for root, _, files in os.walk(REPO_PATH):
        if "/." in root or "/tests" in root or "/__pycache__" in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_chunks = []
            
            if file.endswith(".py"):
                print(f"  [AST] Parsing Python: {file}")
                file_chunks = process_python_file(file_path)
            elif file.endswith(".ipynb"):
                print(f"  [NB]  Parsing Notebook: {file}")
                file_chunks = process_notebook_file(file_path)
            elif file.endswith(".md") or file.endswith(".txt") or file.endswith(".rst"):
                print(f"  [DOC] Reading Document: {file}")
                file_chunks = process_text_file(file_path, "doc")
                
            for chunk in file_chunks:
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                ids.append(str(uuid.uuid4()))

    # Process PDFs
    print("Scanning for PDF files...")
    if os.path.exists(PAPER_PATH):
        for file in os.listdir(PAPER_PATH):
            if file.endswith(".pdf"):
                file_path = os.path.join(PAPER_PATH, file)
                try:
                    reader = pypdf.PdfReader(file_path)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if not text: continue
                        
                        chunk_size = 1000
                        overlap = 200
                        for j in range(0, len(text), chunk_size - overlap):
                            chunk = text[j : j + chunk_size]
                            if len(chunk) < 50: continue
                            
                            documents.append(f"File: {file}\nPage: {i+1}\n\n{chunk}")
                            metadatas.append({"source": file_path, "filename": file, "type": "pdf", "page": i+1})
                            ids.append(str(uuid.uuid4()))
                except Exception as e:
                    print(f"Error processing PDF {file_path}: {e}")

    print(f"Found {len(documents)} chunks. Generating embeddings and storing...")
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"Processed batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")

    print("Knowledge base built successfully!")

if __name__ == "__main__":
    ensure_git_repo(REPO_PATH, "https://github.com/deepmodeling/DeePTB.git")
    build_knowledge_base()
