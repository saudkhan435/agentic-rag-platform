from app.retrieval.search import semantic_search
from app.retrieval.reranker import rerank_results


def retrieve_context(
    query: str,
    candidate_limit: int = 15,
    final_limit: int = 5
) -> list[dict]:

    candidates = semantic_search(
        query=query,
        limit=candidate_limit
    )

    results = rerank_results(
        query=query,
        results=candidates,
        limit=final_limit
    )

    return results