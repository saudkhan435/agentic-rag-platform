import os
import tempfile
import time
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
import logging
from app.database.connection import test_connection
from app.workflow.rag_graph import run_rag_graph
from app.ingestion.service import ingest_pdf
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Agentic RAG Platform",
    version="0.1.0",
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration = time.perf_counter() - start_time

        logger.exception(
            "HTTP request failed | "
            "method=%s | path=%s | duration=%.2fs",
            request.method,
            request.url.path,
            duration
        )

        raise

    duration = time.perf_counter() - start_time

    logger.info(
        "HTTP request completed | "
        "method=%s | path=%s | "
        "status=%d | duration=%.2fs",
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    return response

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