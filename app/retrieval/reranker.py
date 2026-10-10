from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(MODEL_NAME)


def rerank_results(
    query: str,
    results: list[dict],
    limit: int = 5
) -> list[dict]:

    pairs = [
        [query, result["content"]]
        for result in results
    ]

    scores = model.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    ranked_results = sorted(
        results,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return ranked_results[:limit]