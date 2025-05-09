import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import torch
import asyncio

# Устанавливаем маркер окружения для тестов
os.environ["TESTING"] = "True"

# Мок для LLM
@pytest.fixture
def mock_llm():
    """Мокаем LLM функции для тестов."""
    # Ничего не делаем, потому что теперь используем Ollama
    yield

# Мок для RAG-сервиса
@pytest.fixture
def mock_rag():
    mock = MagicMock()
    mock.generate_prompt.return_value = """
    Вопрос: Тестовый вопрос
    
    Контекст:
    Это тестовый контекст для RAG.
    
    Ответь на вопрос, используя информацию из контекста.
    
    ОТВЕТ:
    """
    return mock

# Устанавливаем моки для модулей
@pytest.fixture(autouse=True)
def mock_environment():
    # Убедимся, что get_rag_service вернёт наш мок
    with patch("app.main.get_rag_service") as mock_get_rag:
        mock_rag_service = MagicMock()
        mock_rag_service.generate_prompt.return_value = """
        Вопрос: Тестовый вопрос
        
        Контекст:
        Это тестовый контекст для RAG.
        
        Ответь на вопрос, используя информацию из контекста.
        
        ОТВЕТ:
        """
        mock_get_rag.return_value = mock_rag_service
        
        yield 