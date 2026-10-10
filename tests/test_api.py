from fastapi.testclient import TestClient

from app import main as main_module


client = TestClient(main_module.app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Agentic RAG Platform is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": "   "
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Question cannot be empty"
    }


def test_ask_returns_rag_result(monkeypatch):
    def fake_run_rag_graph(question: str):
        return {
            "answer": "30 LP",
            "sources": [
                {
                    "document": "Info_Slides (1).pdf",
                    "chunk_index": 12
                }
            ]
        }

    monkeypatch.setattr(
        main_module,
        "run_rag_graph",
        fake_run_rag_graph
    )

    response = client.post(
        "/ask",
        json={
            "question": "How many LP are required for the Master Thesis?"
        }
    )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "30 LP",
        "sources": [
            {
                "document": "Info_Slides (1).pdf",
                "chunk_index": 12
            }
        ]
    }