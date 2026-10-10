import logging
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.generation.service import generate_answer
from app.mcp.client import search_documents_via_mcp_sync


load_dotenv()

logger = logging.getLogger(__name__)


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

REWRITE_MODEL = os.getenv(
    "REWRITE_MODEL",
    "llama3.2:3b"
)

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "1")
)

RELEVANCE_THRESHOLD = float(
    os.getenv("RELEVANCE_THRESHOLD", "-3.0")
)


rewrite_llm = ChatOllama(
    model=REWRITE_MODEL,
    temperature=0,
    base_url=OLLAMA_BASE_URL
)


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    context: list[dict]
    answer: str
    sources: list[dict]
    retries: int
    retrieval_good: bool


def retrieve_node(state: RAGState) -> dict:
    search_query = state.get(
        "search_query",
        state["question"]
    )

    logger.info(
        "Retrieval started | query=%r",
        search_query
    )

    results = search_documents_via_mcp_sync(
        query=search_query,
        limit=5
    )

    logger.info(
        "Retrieval completed | results=%d",
        len(results)
    )

    return {
        "context": results
    }


def evaluate_retrieval_node(state: RAGState) -> dict:
    results = state.get("context", [])

    best_score = max(
        (
            result["rerank_score"]
            for result in results
        ),
        default=float("-inf")
    )

    retrieval_good = (
        best_score >= RELEVANCE_THRESHOLD
    )

    logger.info(
        "Retrieval evaluated | "
        "best_score=%.4f | "
        "threshold=%.4f | "
        "good=%s",
        best_score,
        RELEVANCE_THRESHOLD,
        retrieval_good
    )

    return {
        "retrieval_good": retrieval_good
    }


def choose_next_step(
    state: RAGState
) -> Literal[
    "generate",
    "rewrite",
    "insufficient"
]:
    retries = state.get("retries", 0)

    if state["retrieval_good"]:
        logger.info(
            "Routing decision | route=generate | retries=%d",
            retries
        )

        return "generate"

    if retries >= MAX_RETRIES:
        logger.warning(
            "Routing decision | "
            "route=insufficient | retries=%d",
            retries
        )

        return "insufficient"

    logger.info(
        "Routing decision | route=rewrite | retries=%d",
        retries
    )

    return "rewrite"


def rewrite_query_node(state: RAGState) -> dict:
    question = state["question"]
    retries = state.get("retries", 0)

    prompt = f"""
Rewrite the following question into a concise semantic search query.

Rules:
- Preserve the original meaning.
- Preserve important entities and nouns.
- Do not invent new concepts.
- Convert vague wording into clearer wording when possible.
- Prefer a short noun phrase.
- Do not answer the question.
- Return only the rewritten search query.

Question:
{question}

Search query:
"""

    response = rewrite_llm.invoke(prompt)

    rewritten_query = str(response.content).strip()

    logger.info(
        "Query rewritten | "
        "original=%r | "
        "rewritten=%r | "
        "retry=%d",
        question,
        rewritten_query,
        retries + 1
    )

    return {
        "search_query": rewritten_query,
        "retries": retries + 1
    }


def generate_node(state: RAGState) -> dict:
    context = state.get("context", [])

    logger.info(
        "Answer generation started | context_chunks=%d",
        len(context)
    )

    result = generate_answer(
        question=state["question"],
        results=context
    )

    logger.info(
        "Answer generation completed | sources=%d",
        len(result["sources"])
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def insufficient_context_node(
    state: RAGState
) -> dict:
    logger.warning(
        "Retrieval insufficient after retries | "
        "question=%r | retries=%d",
        state["question"],
        state.get("retries", 0)
    )

    return {
        "answer": (
            "The available documents do not "
            "provide enough information."
        ),
        "sources": []
    }


workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "evaluate",
    evaluate_retrieval_node
)

workflow.add_node(
    "rewrite",
    rewrite_query_node
)

workflow.add_node(
    "generate",
    generate_node
)

workflow.add_node(
    "insufficient",
    insufficient_context_node
)


workflow.add_edge(
    START,
    "retrieve"
)

workflow.add_edge(
    "retrieve",
    "evaluate"
)

workflow.add_conditional_edges(
    "evaluate",
    choose_next_step,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "insufficient": "insufficient"
    }
)

workflow.add_edge(
    "rewrite",
    "retrieve"
)

workflow.add_edge(
    "generate",
    END
)

workflow.add_edge(
    "insufficient",
    END
)


rag_graph = workflow.compile()


def run_rag_graph(question: str) -> dict:
    logger.info(
        "RAG workflow started | question=%r",
        question
    )

    result = rag_graph.invoke(
        {
            "question": question,
            "search_query": question,
            "retries": 0
        }
    )

    logger.info(
        "RAG workflow completed | sources=%d",
        len(result.get("sources", []))
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", [])
    }