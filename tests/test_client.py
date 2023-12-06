import pinecone
import pytest
from fastapi.testclient import TestClient

from app.agent import ExecutorAgent
from app.agent.retriever import PineconeRetriever
from app.main import app

client = TestClient(app)


def test_completion(monkeypatch):
    def mockreturn(*args, **kwargs):
        return "Hello"

    monkeypatch.setattr(ExecutorAgent, "run", mockreturn)
    req_body = {"query": "Hi"}
    response = client.post("/api/v1/chat/completion", json=req_body)
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "Hello" == response.json()["answer"]


def test_similarity_search(monkeypatch):
    def mockreturn(*args, **kwargs):
        return "Hello"

    monkeypatch.setattr(PineconeRetriever, "get_relevant_doc_string", mockreturn)
    monkeypatch.setattr(PineconeRetriever, "get_pinecone_index", mockreturn)
    monkeypatch.setattr(pinecone, "init", mockreturn)

    response = client.get("/api/v1/chat/similarity-search?query=Hi")
    assert response.status_code == 200
    assert isinstance(response.text, str)
    assert '"Hello"' == response.text


@pytest.mark.live
def test_live_completion():
    req_body = {"query": "Hi"}
    response = client.post("/api/v1/chat/completion", json=req_body)
    assert response.status_code == 200


@pytest.mark.live
def test_live_similarity_search():
    response = client.get("/api/v1/chat/similarity-search?query=Hi")
    assert response.status_code == 200
