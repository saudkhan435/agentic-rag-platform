import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.database.connection import test_connection
from app.workflow.rag_graph import run_rag_graph
from app.ingestion.service import ingest_pdf


app = FastAPI(
    title="Agentic RAG Platform",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    return run_rag_graph(question)


@app.get("/")
def root():
    return {
        "message": "Agentic RAG Platform is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/db-health")
def database_health():
    result = test_connection()

    return {
        "database": "connected",
        "test_result": result
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    temp_path = None

    try:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        result = ingest_pdf(
            file_path=temp_path,
            original_filename=file.filename
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)