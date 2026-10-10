import logging
import time

from app.retrieval.reranker import rerank_results
from app.retrieval.search import semantic_search


logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    candidate_limit: int = 15,
    final_limit: int = 5
) -> list[dict]:
    logger.info(
        "Retrieval pipeline started | "
        "candidate_limit=%d | final_limit=%d",
        candidate_limit,
        final_limit
    )

    retrieval_start = time.perf_counter()

    candidates = semantic_search(
        query=query,
        limit=candidate_limit
    )

    retrieval_duration = (
        time.perf_counter() - retrieval_start
    )

    logger.info(
        "Vector search completed | "
        "candidates=%d | duration=%.2fs",
        len(candidates),
        retrieval_duration
    )

    rerank_start = time.perf_counter()

    results = rerank_results(
        query,
        candidates,
        limit=final_limit
    )

    rerank_duration = (
        time.perf_counter() - rerank_start
    )

    logger.info(
        "Reranking completed | "
        "results=%d | duration=%.2fs",
        len(results),
        rerank_duration
    )

    total_duration = (
        time.perf_counter() - retrieval_start
    )

    logger.info(
        "Retrieval pipeline completed | "
        "duration=%.2fs",
        total_duration
    )

    return results