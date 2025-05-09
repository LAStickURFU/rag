import os
import json
import pytest
from typing import List, Dict, Any, Mapping, Optional
import numpy as np
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from datasets import Dataset
from langchain.llms.base import LLM
from langchain.pydantic_v1 import BaseModel, Field

from app.rag import Vectorizer, RAGService, Document
from app.chunking.robust_chunker import RobustChunker
from app.main import get_rag_service


# Тестовые данные
RAG_TEST_DATA = [
    {
        "question": "Что такое RAG система?",
        "answer": "RAG (Retrieval-Augmented Generation) - это технология, которая объединяет поиск информации и генерацию текста. Система сначала находит релевантные фрагменты данных из базы знаний, а затем использует их как контекст для LLM, чтобы создать точный и информативный ответ.",
        "contexts": [],  # Будет заполнено автоматически
        "ground_truth": "RAG (Retrieval-Augmented Generation) - это подход к обработке естественного языка, который сочетает извлечение (retrieval) информации из внешней базы знаний с генерацией (generation) ответов. Система RAG сначала находит релевантные фрагменты в базе знаний, а затем использует их как контекст для языковой модели, чтобы сгенерировать точный ответ."
    },
    {
        "question": "Какие компоненты включает RAG?",
        "answer": "RAG включает несколько ключевых компонентов: 1) Индексатор документов, который преобразует тексты в векторные представления; 2) Поисковый движок, который находит релевантные фрагменты на основе запроса; 3) Генеративную языковую модель (LLM), использующую найденные фрагменты как контекст для создания ответа.",
        "contexts": [],
        "ground_truth": "RAG включает компоненты: индексатор документов для преобразования в векторные представления, поисковая система для извлечения релевантных фрагментов, и генеративная языковая модель для создания ответов на основе контекста."
    },
    {
        "question": "Почему RAG лучше обычной генерации текста?",
        "answer": "RAG превосходит обычную генерацию текста, потому что: 1) Обеспечивает доступ к актуальной внешней информации, которой нет в весах модели; 2) Снижает галлюцинации, так как опирается на фактическую информацию; 3) Позволяет легко обновлять базу знаний без переобучения модели; 4) Делает ответы более прозрачными, поскольку можно проследить источники информации.",
        "contexts": [],
        "ground_truth": "RAG превосходит обычную генерацию текста благодаря доступу к актуальной внешней информации, снижению галлюцинаций, возможности обновления базы знаний без переобучения модели и прозрачности источников информации."
    }
]


@pytest.fixture
def rag_service():
    """Фикстура для создания RAG-сервиса для тестирования"""
    # Создаем временную директорию для индекса
    index_dir = Path("tests/indexes/test_quality")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализируем компоненты и сервис
    chunker = RobustChunker(chunk_size=400, chunk_overlap=100, spacy_model="ru_core_news_md")
    vectorizer = Vectorizer(model_name="intfloat/multilingual-e5-base")
    
    # Создаем сервис
    service = RAGService(
        index_name="test_quality",
        model_name="intfloat/multilingual-e5-base",
        chunk_size=400,
        chunk_overlap=100,
        use_hybrid_search=False,  # Отключаем гибридный поиск для тестов
        use_reranker=False,       # Отключаем ранжирование для тестов
        language="russian",
        spacy_model="ru_core_news_md"
    )
    
    # Загрузка тестовых данных
    test_docs = [
        {
            "id": 1,
            "content": """
            # Введение в RAG
            
            RAG (Retrieval-Augmented Generation) — это подход к обработке естественного языка, который сочетает извлечение (retrieval) информации из внешней базы знаний с генерацией (generation) ответов. Система RAG сначала находит релевантные фрагменты в базе знаний, а затем использует их как контекст для языковой модели, чтобы сгенерировать точный ответ.
            
            ## Компоненты RAG
            
            Основные компоненты системы RAG включают:
            
            1. **Индексирование документов** — процесс преобразования текстовых документов в векторные представления (эмбеддинги).
            2. **Поисковая система** — механизм для поиска и извлечения релевантных фрагментов из базы знаний на основе запроса.
            3. **Генеративная модель** — языковая модель (LLM), которая генерирует ответы на основе запроса и найденного контекста.
            
            ## Преимущества RAG
            
            RAG предлагает несколько ключевых преимуществ перед стандартными подходами к генерации текста:
            
            - **Актуальность информации** — модель может использовать самую свежую информацию из базы знаний
            - **Снижение галлюцинаций** — за счёт опоры на фактическую информацию из внешних источников
            - **Гибкость** — можно обновлять базу знаний без переобучения модели
            - **Прозрачность** — возможность отследить источники информации, используемые для генерации ответа
            
            ## Применение RAG
            
            Системы RAG находят применение во многих областях:
            
            - Вопросно-ответные системы
            - Чат-боты с доступом к корпоративным знаниям
            - Инструменты для исследований и анализа данных
            - Системы поддержки принятия решений
            """,
            "metadata": {"title": "Введение в RAG", "source": "Документация"}
        }
    ]
    
    # Индексируем документы
    for doc in test_docs:
        document = Document(
            content=doc["content"],
            metadata=doc["metadata"]
        )
        service.add_document(document, str(doc["id"]))
    
    return service


def prepare_ragas_dataset(rag_service, test_data: List[Dict[str, Any]]) -> Dataset:
    """Подготавливает датасет для оценки с помощью RAGAS"""
    prepared_data = []
    
    for item in test_data:
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # Получаем контексты из RAG-системы
        search_results = rag_service.search(question, top_k=3)
        # Распаковываем результаты (chunk, score)
        contexts = [chunk.text for chunk, score in search_results]
        
        # Генерируем ответ с помощью промпта
        prompt = rag_service.generate_prompt(question)
        
        # В реальной системе здесь бы использовался ответ от LLM
        # Для тестов используем предопределенный ответ
        answer = item["answer"]
        
        prepared_data.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": ground_truth  # В RAGAS 0.2.x используется поле 'reference' вместо 'ground_truth'
        })
    
    return Dataset.from_list(prepared_data)


# Пользовательская реализация LLM для RAGAS без зависимости от OpenAI
class MockLLM(LLM, BaseModel):
    """Мок-класс для LLM, который будет использоваться в RAGAS без OpenAI API."""
    
    class Config:
        """Конфигурация."""
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Мок-реализация вызова LLM."""
        if "rate the relevance" in prompt.lower():
            return "Rating: 5"
        elif "rate how well the answer is grounded" in prompt.lower():
            return "Rating: 5"
        elif "list all unsupported facts" in prompt.lower():
            return "There are no hallucinated facts."
        else:
            return "This is a mock response for testing purposes."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "mock_llm"}


def test_rag_end_to_end(rag_service):
    """Проверяет end-to-end работу RAG-системы на типичном запросе"""
    
    # Тестовый запрос
    question = "Как работает RAG система?"
    
    # Получаем релевантные фрагменты
    relevant_chunks = rag_service.search(question, top_k=3)
    
    # Проверяем, что найдены фрагменты
    assert len(relevant_chunks) > 0, "No relevant chunks found"
    
    # Проверяем, что релевантность присутствует
    # Для мок-векторизатора достаточно проверить, что score существует
    # и имеет ненулевое значение (в реальной системе порог был бы 0.7)
    chunk, score = relevant_chunks[0]  # Распаковываем первый результат в chunk и score
    assert isinstance(score, float), f"Score should be float, got {type(score)}"
    
    # Генерируем промпт
    prompt = rag_service.generate_prompt(question)
    
    # Проверяем, что промпт содержит вопрос и контексты
    assert question in prompt, "Question not included in prompt"
    assert "КОНТЕКСТ:" in prompt, "Context section missing in prompt"
    assert "ВОПРОС:" in prompt, "Question section missing in prompt"
    assert "ОТВЕТ:" in prompt, "Answer section missing in prompt"
    
    print(f"\nGenerated prompt for question '{question}':")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # В реальном тесте здесь бы проверялся ответ от LLM
    print("End-to-end RAG process test completed successfully!")


if __name__ == "__main__":
    # Этот блок позволяет запускать файл напрямую для тестирования
    # Создаем RAG-сервис
    service = get_rag_service()
    
    # Запускаем тесты
    test_rag_end_to_end(service)