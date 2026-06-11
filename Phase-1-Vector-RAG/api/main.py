from fastapi import FastAPI, UploadFile, File, Body
from langchain_core.messages import HumanMessage, AIMessage
from api.core.processor import process_document
from api.core.engine import LegalEngine
from langchain_community.vectorstores import Chroma
import shutil
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()
engine = LegalEngine()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks = process_document(file_location)
    # Add to Chroma
    engine.db.add_documents(chunks)
    return {"message": f"File {file.filename} processed and indexed."}

chat_memories = {}

@app.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question")
    session_id = payload.get("session_id", "default")
    
    # Initialize history for new sessions
    if session_id not in chat_memories:
        chat_memories[session_id] = []

    graph = engine.build_graph()
    inputs = {
        "question": question,
        "history": chat_memories[session_id]
    }
    
    result = graph.invoke(inputs)
    
    # Update local memory
    chat_memories[session_id] = result["history"]
    
    return {"answer": result["answer"]}
