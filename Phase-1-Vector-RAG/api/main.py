import os
import shutil
from fastapi import FastAPI, UploadFile, File, Body, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from api.core.processor import process_document
from api.core.engine import LegalEngine

load_dotenv()
app = FastAPI()
engine = LegalEngine()

templates = Jinja2Templates(directory="gui/templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={}
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks = process_document(file_location)
    engine.db.add_documents(chunks)
    return {"message": f"File {file.filename} indexed successfully."}

@app.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question")
    # LangGraph invocation
    graph = engine.build_graph()
    result = graph.invoke({"question": question, "history": []})
    return {"answer": result["answer"]}
