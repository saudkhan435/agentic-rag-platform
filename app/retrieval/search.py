from sqlalchemy import select

from app.database.connection import SessionLocal
from app.database.models import Document, DocumentChunk
from app.embeddings.model import generate_embedding


def semantic_search(query: str, limit: int = 5):
    query_embedding = generate_embedding(query)

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    )

    statement = (
        select(
            DocumentChunk,
            Document,
            distance.label("distance")
        )
        .join(
            Document,
            DocumentChunk.document_id == Document.id
        )
        .where(
            DocumentChunk.embedding.is_not(None)
        )
        .order_by(distance)
        .limit(limit)
    )

    with SessionLocal() as session:
        rows = session.execute(statement).all()

        results = []

        for chunk, document, distance_value in rows:
            results.append(
                {
                    "document": document.filename,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "distance": float(distance_value)
                }
            )

        return results