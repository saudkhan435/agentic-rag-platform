from app.workflow.rag_graph import choose_next_step


def test_good_retrieval_goes_to_generate():
    state = {
        "retrieval_good": True,
        "retries": 0
    }

    assert choose_next_step(state) == "generate"


def test_bad_retrieval_goes_to_rewrite():
    state = {
        "retrieval_good": False,
        "retries": 0
    }

    assert choose_next_step(state) == "rewrite"


def test_bad_retrieval_after_retry_goes_to_insufficient():
    state = {
        "retrieval_good": False,
        "retries": 1
    }

    assert choose_next_step(state) == "insufficient"