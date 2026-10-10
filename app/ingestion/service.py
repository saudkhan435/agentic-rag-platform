from pathlib import Path

from app.database.connection import SessionLocal
from app.database.models import Document, DocumentChunk
from app.embeddings.model import generate_embeddings
from app.ingestion.pdf_loader import extract_text_from_pdf
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import split_text


def ingest_pdf(
    file_path: str,
    original_filename: str | None = None
):
    text = extract_text_from_pdf(file_path)

    cleaned_text = clean_text(text)

    chunks = split_text(cleaned_text)

    embeddings = generate_embeddings(chunks)

    filename = original_filename or Path(file_path).name

    with SessionLocal() as session:
        document = Document(
            filename=filename,
            content_type="application/pdf"
        )

        session.add(document)
        session.flush()

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            document_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding
            )

            session.add(document_chunk)

        session.commit()

        return {
            "document_id": document.id,
            "filename": document.filename,
            "chunks_created": len(chunks)
        }