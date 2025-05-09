"""
Тесты для проверки работы модуля Ollama через прямые вызовы, 
без использования TestClient.
"""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock

from app.ollama_client import OllamaLLM

# Тест для прямой проверки работы OllamaLLM
@pytest.mark.asyncio
async def test_ollama_client_direct():
    """Тестирует прямую работу с классом OllamaLLM"""
    
    # Моканье клиента Ollama
    with patch("ollama.AsyncClient.generate") as mock_generate:
        # Настраиваем возвращаемое значение
        mock_generate.return_value = {"response": "Тестовый ответ от OllamaLLM"}
        
        # Создаем экземпляр и вызываем метод
        client = OllamaLLM("test-model")
        client._model_loaded = True
        
        # Вызываем метод генерации
        response = await client.generate("Тестовый вопрос")
        
        # Проверяем результат
        assert response == "Тестовый ответ от OllamaLLM"
        assert mock_generate.called

# Тест для проверки обработки ошибок
@pytest.mark.asyncio
async def test_ollama_error_handling_direct():
    """Тестирует обработку ошибок в OllamaLLM"""
    
    # Моканье клиента Ollama с вызовом исключения
    with patch("ollama.AsyncClient.generate") as mock_generate:
        mock_generate.side_effect = Exception("Тестовая ошибка")
        
        # Создаем экземпляр и вызываем метод
        client = OllamaLLM("test-model")
        client._model_loaded = True
        
        # Вызываем метод генерации
        response = await client.generate("Тестовый вопрос")
        
        # Проверяем результат
        assert "Ошибка генерации ответа" in response
        assert "Тестовая ошибка" in response

# Тест для проверки переключения на доступную модель
@pytest.mark.asyncio
async def test_ollama_fallback_model():
    """Тестирует использование доступной модели при ошибке загрузки основной"""
    
    # Моканье методов
    with patch("ollama.AsyncClient.list") as mock_list, \
         patch("ollama.AsyncClient.generate") as mock_generate:
        
        # Настройка возвращаемых данных для списка моделей
        mock_list.return_value = {
            "models": [
                {"name": "fallback-model", "model": "fallback-model"}
            ]
        }
        
        # Настройка возвращаемых данных для генерации
        mock_generate.return_value = {"response": "Ответ от запасной модели"}
        
        # Создаем клиент с несуществующей моделью
        client = OllamaLLM("non-existent-model")
        
        # Вызываем метод генерации
        response = await client.generate("Тестовый вопрос")
        
        # Проверяем, что используется запасная модель
        assert "fallback-model" == client.model_name
        assert "Ответ от запасной модели" == response

# Запуск тестов при прямом вызове файла
if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 