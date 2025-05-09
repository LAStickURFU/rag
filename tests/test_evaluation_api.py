import os
os.environ["DATABASE_URL"] = "postgresql+psycopg2://raguser:ragpass@localhost:5432/ragdb_test"
os.environ["TESTING"] = "True"
os.environ["OPENAI_API_KEY"] = "sk-test-mock-key"
os.environ["USE_OPENAI"] = "False"
import pytest
from fastapi.testclient import TestClient
from app.main import app
import json
import pathlib
from sqlalchemy import create_engine
from app.database import Base

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

client = TestClient(app)

@pytest.fixture(scope="module")
def auth_headers():
    client.post("/test/init-db")
    username = "apitestuser"
    password = "apitestpass"
    client.post("/register", json={"username": username, "password": password})
    token_resp = client.post("/token", data={"username": username, "password": password})
    token = token_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# def test_evaluate_rag_valid(auth_headers):
#     # Минимальный валидный запрос
#     payload = {
#         "eval_items": [
#             {
#                 "question": "Что такое RAG?",
#                 "answer": "RAG — это retrieval-augmented generation.",
#                 "ground_truth": "RAG — это подход, сочетающий поиск и генерацию."
#             }
#         ]
#     }
#     response = client.post("/api/evaluate-rag", json=payload, headers=auth_headers)
#     assert response.status_code == 200
#     data = response.json()
#     assert "metrics" in data
#     assert "evaluation_file" in data
#     assert isinstance(data["metrics"], dict)


def test_evaluate_rag_invalid(auth_headers):
    # Пустой запрос
    payload = {"eval_items": []}
    response = client.post("/api/evaluate-rag", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "No evaluation items provided" in response.text


def test_last_results_not_found(tmp_path, monkeypatch, auth_headers):
    # Переопределяем директорию результатов на пустую
    monkeypatch.setattr(pathlib, "Path", lambda *a, **kw: tmp_path)
    response = client.get("/api/evaluation/last_results", headers=auth_headers)
    assert response.status_code == 404
    assert "Нет сохранённых результатов оценки" in response.text


def test_last_results_found(tmp_path, monkeypatch, auth_headers):
    # Создаём фейковый результат
    results = {"metrics": {"faithfulness": 1.0}, "meta": {}}
    file = tmp_path / "rag_eval_results_123.json"
    file.write_text(json.dumps(results), encoding="utf-8")
    monkeypatch.setattr(pathlib, "Path", lambda *a, **kw: tmp_path)
    response = client.get("/api/evaluation/last_results", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["metrics"]["faithfulness"] == 1.0 