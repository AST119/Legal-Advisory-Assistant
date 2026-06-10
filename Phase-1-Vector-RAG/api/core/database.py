import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the storage path for the persistent vector database
VECTOR_DB_DIR = "vector_db"

def get_embeddings():
    """
    Standard industry embedding model. 
    'all-MiniLM-L6-v2' is fast and runs locally.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    """
    Returns a ChromaDB instance. 
    If the database exists on disk, it loads it; otherwise, it initializes it.
    """
    # Ensure the directory exists
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)
        
    embeddings = get_embeddings()
    
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name="legal_docs"
    )