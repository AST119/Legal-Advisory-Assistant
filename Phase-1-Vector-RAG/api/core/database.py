import os
from langchain_chroma import Chroma  # Updated
from langchain_huggingface import HuggingFaceEmbeddings  # Updated

VECTOR_DB_DIR = "vector_db"

def get_embeddings():
    """
    Standard industry embedding model. 
    'all-MiniLM-L6-v2' is fast and runs locally.
    """
    # Uses the updated langchain-huggingface class
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    """
    If the database exists on disk, it loads it; otherwise, it initializes it.
    """
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)
        
    embeddings = get_embeddings()
    
    # Uses the updated langchain-chroma class
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name="legal_docs"
    )
