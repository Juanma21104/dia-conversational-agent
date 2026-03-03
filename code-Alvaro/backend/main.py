# backend/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from backend.rag import BasicRAG

app = FastAPI()
rag = BasicRAG()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    saved_paths = []
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(file_path)

    # Indexamos solo los archivos recién subidos
    rag.add_documents_from_files(saved_paths)

    return {"status": "ok"}


@app.post("/chat")
async def chat(payload: dict):

    question = payload.get("question")
    selected_files = payload.get("selected_files")

    if not question or not selected_files:
        return {"answer": "Missing question or selected_files"}

    answer = rag.query(question, selected_files)

    return {"answer": answer}


@app.get("/list_documents")
async def list_documents():

    documents = rag.list_documents()

    return {
        "documents": documents
    }