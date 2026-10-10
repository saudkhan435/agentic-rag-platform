import os

from dotenv import load_dotenv

from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.generation.service import generate_answer
from app.mcp.client import search_documents_via_mcp_sync

load_dotenv()

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
    temperature=0
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

    results = search_documents_via_mcp_sync(
        query=search_query,
        limit=5
    )

    return {
        "context": results
    }


def evaluate_retrieval_node(state: RAGState) -> dict:
    context = state.get("context", [])

    if not context:
        return {
            "retrieval_good": False
        }

    best_score = max(
        result["rerank_score"]
        for result in context
    )

    retrieval_good = (
        best_score >= RELEVANCE_THRESHOLD
    )

    print(
        f"Best rerank score: {best_score:.4f}"
    )
    print(
        f"Retrieval good: {retrieval_good}"
    )

    return {
        "retrieval_good": retrieval_good
    }


def choose_next_step(
    state: RAGState
) -> Literal["generate", "rewrite", "insufficient"]:

    retries = state.get("retries", 0)

    if state["retrieval_good"]:
        return "generate"

    if retries >= MAX_RETRIES:
        return "insufficient"

    return "rewrite"

def insufficient_context_node(
    state: RAGState
) -> dict:
    return {
        "answer": (
            "The available documents do not provide "
            "enough information."
        ),
        "sources": []
    }

def rewrite_query_node(state: RAGState) -> dict:
    prompt = f"""
Rewrite the user's question into a concise search query
for semantic document retrieval.

Rules:
1. Preserve the original meaning.
2. Preserve important entities and nouns from the question.
3. Do not invent new concepts that the user did not mention.
4. Convert vague wording into clearer retrieval terminology.
5. Prefer a short noun phrase rather than a full sentence.
6. Do not answer the question.
7. Return only the rewritten query.

Original question:
{state["question"]}
"""

    response = rewrite_llm.invoke(prompt)

    rewritten_query = response.content.strip()

    print(
        f"Rewritten query: {rewritten_query}"
    )

    return {
        "search_query": rewritten_query,
        "retries": state.get("retries", 0) + 1
    }


def generate_node(state: RAGState) -> dict:
    result = generate_answer(
        question=state["question"],
        results=state["context"]
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


workflow = StateGraph(RAGState)

workflow.add_node(
    "retrieve",
    retrieve_node
)

workflow.add_node(
    "evaluate_retrieval",
    evaluate_retrieval_node
)

workflow.add_node(
    "rewrite_query",
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
    "evaluate_retrieval"
)

workflow.add_conditional_edges(
    "evaluate_retrieval",
    choose_next_step,
    {
        "generate": "generate",
        "rewrite": "rewrite_query",
        "insufficient": "insufficient"
    }
)

workflow.add_edge(
    "rewrite_query",
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
    result = rag_graph.invoke(
        {
            "question": question,
            "search_query": question,
            "retries": 0
        }
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }