import os

from dotenv import load_dotenv

import re

from langchain_ollama import ChatOllama

from app.retrieval.service import retrieve_context

load_dotenv()

MODEL_NAME = os.getenv(
    "GENERATION_MODEL",
    "qwen2.5:7b"
)

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0
)


def generate_answer(
    question: str,
    results: list[dict]
) -> dict:

    context_blocks = []
    sources = []

    for result in results:
        source_name = (
            f"{result['document']} "
            f"(chunk {result['chunk_index']})"
        )

        context_blocks.append(
            f"[Source: {source_name}]\n"
            f"{result['content']}"
        )

        sources.append(
            {
                "document": result["document"],
                "chunk_index": result["chunk_index"]
            }
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a document-grounded question-answering assistant.

Answer the user's question using only the provided context.

Rules:
1. Include only information that directly answers the question.
2. Treat explicit label-value pairs, requirement lists, and structured
   fragments in extracted PDF/OCR text as valid evidence.
3. If the same fact appears consistently in multiple context chunks,
   you may use that as strong supporting evidence.
4. Do not reject an answer only because the extracted text is
   grammatically fragmented or comes from OCR.
5. Do not turn unrelated nearby information into part of the answer.
6. Do not use outside knowledge.
7. If the context truly does not contain enough information, say:
   "The available documents do not provide enough information."
8. Cite the supporting source after each factual claim using the
   exact source label provided in the context.

Question:
{question}

Context:
{context}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content

    cited_chunk_numbers = {
        int(chunk_number)
        for chunk_number in re.findall(
            r"\(chunk\s+(\d+)\)",
            answer
        )
    }

    cited_sources = [
        source
        for source in sources
        if source["chunk_index"] in cited_chunk_numbers
    ]

    return {
        "answer": answer,
        "sources": cited_sources
    }


def answer_question(question: str) -> dict:
    results = retrieve_context(question)

    return generate_answer(
        question=question,
        results=results
    )