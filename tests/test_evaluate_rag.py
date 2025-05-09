"""
Тесты для модуля evaluate_rag.py
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.evaluate_rag import RAGEvaluator, get_builtin_datasets


@pytest.fixture
def mock_rag_service():
    """Создает мок для RAGService"""
    mock = MagicMock()
    mock.search.return_value = [
        (MagicMock(text="Это тестовый контекст 1", doc_id="1", chunk_id="1", metadata={}), 0.95),
        (MagicMock(text="Это тестовый контекст 2", doc_id="2", chunk_id="2", metadata={}), 0.85),
    ]
    mock.generate_prompt.return_value = "Вопрос: Тестовый вопрос\nКонтексты: [текст контекстов]\nОтвет:"
    return mock


@pytest.fixture
def mock_ollama_client():
    """Создает мок для OllamaLLM"""
    mock = MagicMock()
    mock.generate.return_value = "Это тестовый ответ модели"
    return mock


@pytest.fixture
def test_data():
    """Тестовые данные для оценки RAG"""
    return [
        {
            "question": "Тестовый вопрос 1?",
            "answer": "Тестовый ответ 1",
            "ground_truth": "Эталонный ответ 1"
        },
        {
            "question": "Тестовый вопрос 2?",
            "answer": "Тестовый ответ 2",
            "ground_truth": "Эталонный ответ 2"
        }
    ]


@pytest.fixture
def evaluator(mock_rag_service, mock_ollama_client):
    """Создает экземпляр RAGEvaluator с моками"""
    with patch('evaluate_rag.RAGService', return_value=mock_rag_service), \
         patch('evaluate_rag.Ollama', return_value=mock_ollama_client), \
         patch('evaluate_rag.HuggingFaceEmbeddings'):
        evaluator = RAGEvaluator(
            ollama_model_name="test-model",
            embedding_model_name="test-embeddings"
        )
        evaluator.results_dir = Path("tests/test_results")
        evaluator.results_dir.mkdir(exist_ok=True, parents=True)
        yield evaluator
        # Cleanup
        if evaluator.results_dir.exists():
            for f in evaluator.results_dir.glob("*"):
                f.unlink()
            evaluator.results_dir.rmdir()


@pytest.mark.asyncio
async def test_prepare_evaluation_dataset(evaluator, test_data):
    """Тест для метода prepare_evaluation_dataset"""
    dataset = await evaluator.prepare_evaluation_dataset(test_data)
    assert dataset is not None
    assert hasattr(dataset, "column_names")
    assert "user_input" in dataset.column_names
    assert "retrieved_contexts" in dataset.column_names
    assert "response" in dataset.column_names
    assert "reference" in dataset.column_names


@patch('evaluate_rag.evaluate', return_value={"context_precision": 0.85, "faithfulness": 0.92})
@pytest.mark.asyncio
async def test_run_evaluation(mock_evaluate, evaluator, test_data):
    """Тест для метода run_evaluation"""
    results = await evaluator.run_evaluation(test_data)
    assert results is not None
    assert isinstance(results, dict)
    assert "context_precision" in results
    assert "faithfulness" in results
    assert results["context_precision"] == 0.85
    assert results["faithfulness"] == 0.92
    mock_evaluate.assert_called_once()


def test_save_results(evaluator, test_data):
    """Тест для метода save_results"""
    results = {"context_precision": 0.85, "faithfulness": 0.92}
    output_path = evaluator.save_results(results, test_data, "test-dataset")
    assert output_path is not None
    assert Path(output_path).exists()
    
    # Проверяем содержимое файла
    with open(output_path, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert "metrics" in saved_data
    assert saved_data["metrics"]["context_precision"] == 0.85
    assert saved_data["metrics"]["faithfulness"] == 0.92
    assert "dataset_name" in saved_data
    assert saved_data["dataset_name"] == "test-dataset"


def test_visualize_metrics(evaluator):
    """Тест для метода visualize_metrics"""
    results = {"context_precision": 0.85, "faithfulness": 0.92}
    plot_path = evaluator.visualize_metrics(results)
    assert plot_path is not None
    assert Path(plot_path).exists()
    assert plot_path.endswith('.png')


def test_get_builtin_datasets():
    """Тест для функции get_builtin_datasets"""
    with patch('evaluate_rag.Path.mkdir'), \
         patch('evaluate_rag.Path.exists', return_value=True), \
         patch('evaluate_rag.load_dataset'), \
         patch('evaluate_rag.requests.get'):
        datasets = get_builtin_datasets()
        assert datasets is not None
        assert isinstance(datasets, dict) 