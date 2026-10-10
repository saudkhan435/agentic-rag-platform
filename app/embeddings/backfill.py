from sqlalchemy import select

from app.database.connection import SessionLocal
from app.database.models import DocumentChunk
from app.embeddings.model import generate_embeddings


def backfill_embeddings():
    with SessionLocal() as session:

        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.id)
        )

        chunks = session.scalars(statement).all()

        if not chunks:
            print("No chunks need embeddings.")
            return

        texts = [chunk.content for chunk in chunks]

        embeddings = generate_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        session.commit()

        print(f"Stored embeddings for {len(chunks)} chunks.")


if __name__ == "__main__":
    backfill_embeddings()