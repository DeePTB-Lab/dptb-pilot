import os
import chromadb
from sentence_transformers import SentenceTransformer
from dptb_pilot.tools.init import mcp

# Configuration (must match builder)
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")
COLLECTION_NAME = "deeptb_knowledge"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Global resources (lazy loaded)
_client = None
_collection = None
_model = None

def get_resources():
    global _client, _collection, _model
    if _client is None:
        try:
            _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            _collection = _client.get_collection(name=COLLECTION_NAME)
            _model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL, local_files_only=True)
        except Exception as e:
            print(f"Error initializing RAG resources: {e}")
            return None, None, None
    return _client, _collection, _model

@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Search the DeePTB knowledge base (source code, docs, papers) for relevant information.
    Use this tool to find where to look before reading specific files.
    
    Args:
        query: The search query (e.g., "how to calculate band structure").
        n_results: Number of results to return (default: 5).
        
    Returns:
        A string containing relevant text chunks and their source filenames.
    """
    client, collection, model = get_resources()
    if collection is None:
        return "Error: Knowledge base not initialized. Please run build_knowledge_base.py first."

    try:
        query_embedding = model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        output = f"Search results for '{query}':\n\n"
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant results found."

        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            source = metadata.get('filename', 'unknown')
            doc_type = metadata.get('type', 'unknown')
            
            output += f"--- Result {i+1} ({doc_type}: {source}) ---\n"
            output += doc + "\n\n"
            
        return output
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
