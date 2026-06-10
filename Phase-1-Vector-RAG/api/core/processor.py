import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path: str):
    """
    Loads PDF or DOCX and splits them into chunks with metadata attribution.
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Load documents
    documents = loader.load()
    
    # Ensure metadata contains the source filename for attribution
    for doc in documents:
        doc.metadata["source"] = file_name 

    # Split documents into chunks
    # RecursiveCharacterTextSplitter preserves the metadata from the parent 'doc'
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully processed {file_name} into {len(chunks)} chunks.")
    return chunks