#!/usr/bin/env python3
"""
Скрипт для подготовки тестовых данных перед оценкой качества RAG-системы.
Индексирует тестовые документы в Qdrant для обеспечения корректной работы оценки.
"""

import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_test_data(collection_name: str = "test_documents", clear_existing: bool = True):
    """
    Подготавливает тестовые данные для оценки качества RAG-системы.
    
    Args:
        collection_name: Название коллекции в Qdrant для тестовых данных
        clear_existing: Очистить существующую коллекцию перед добавлением
    """
    try:
        # Импортируем необходимые модули
        from app.chunking.robust_chunker import Document
        from app.config import create_rag_service_from_config, RagConfig
        
        # Создаем пользовательскую конфигурацию
        custom_config = RagConfig()
        custom_config.collection_name = collection_name
        
        # Создаем RAG-сервис
        logger.info(f"Initializing RAG service with collection '{collection_name}'")
        rag_service = create_rag_service_from_config(custom_config)
        
        # Если нужно, очищаем существующую коллекцию
        if clear_existing:
            logger.info(f"Clearing existing collection '{collection_name}'")
            # Очищаем индекс
            if hasattr(rag_service, "index") and rag_service.index:
                rag_service.index.clear()
            else:
                logger.warning("Could not access index directly. Trying alternative approach.")
                from qdrant_client import QdrantClient
                from app.config import QdrantConfig
                qdrant_config = QdrantConfig()
                client = QdrantClient(
                    host=qdrant_config.host,
                    port=qdrant_config.port
                )
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"Collection '{collection_name}' deleted")
                    # Воссоздаем RAG-сервис после удаления коллекции
                    rag_service = create_rag_service_from_config(custom_config)
                except Exception as e:
                    logger.warning(f"Error deleting collection: {e}")
        
        # Создаем тестовые документы для разных тем
        documents = [
            # RAG основы
            Document(
                content="""
                RAG (Retrieval Augmented Generation) — это подход, при котором генеративная языковая модель дополняется системой поиска.
                Сначала находятся релевантные документы, затем на их основе модель генерирует точный и информативный ответ.
                Архитектура RAG-систем позволяет решить несколько ключевых проблем современных языковых моделей:
                1. Ограниченность знаний: Предварительно обученные модели имеют фиксированный набор знаний.
                2. Невозможность обновления: Трудно обновлять знания в языковой модели без переобучения.
                3. Галлюцинации: Ситуации, когда модель генерирует правдоподобный, но фактически неверный контент.
                """,
                metadata={"title": "Основы RAG", "source": "rag_basics.md", "category": "theory"}
            ),
            
            # Компоненты RAG
            Document(
                content="""
                Улучшенная RAG-система включает следующие компоненты:
                - Гибридный поиск (Dense + Sparse Retrieval): комбинация семантического и ключевого поиска
                - Переранжирование с CrossEncoder: уточнение релевантности найденных документов
                - Улучшенный чанкинг: интеллектуальное разбиение документов на фрагменты
                - Адаптивный выбор количества документов для контекста: динамическое определение оптимального числа фрагментов
                """,
                metadata={"title": "Компоненты RAG", "source": "rag_components.md", "category": "architecture"}
            ),
            
            # Токеновый чанкер
            Document(
                content="""
                Token-based чанкер разбивает документы на фрагменты фиксированного размера в токенах, 
                что позволяет более эффективно использовать контекстное окно модели и улучшить качество ответов.
                Преимущества token-based чанкера:
                1. Сокращение количества чанков на 80-90%
                2. Увеличение среднего размера чанка в токенах на 300-400%
                3. Повышение эффективности использования контекстного окна
                4. Ускорение обработки на 95-98%
                """,
                metadata={"title": "Токеновый чанкер", "source": "token_chunker.md", "category": "implementation"}
            ),
            
            # Методы оценки RAG
            Document(
                content="""
                Для оценки эффективности RAG-систем используются специальные метрики:
                1. Контекстная релевантность (Context Precision): насколько предоставленный контекст соответствует запросу
                2. Полнота информации (Context Recall): содержит ли найденный контекст всю необходимую информацию
                3. Точность ответа (Answer Similarity): соответствие сгенерированного ответа истине
                4. Своевременность (Latency): скорость работы системы от запроса до ответа
                """,
                metadata={"title": "Методы оценки RAG", "source": "rag_evaluation.md", "category": "evaluation"}
            )
        ]
        
        # Индексируем документы
        logger.info(f"Indexing {len(documents)} test documents")
        for i, doc in enumerate(documents):
            doc_id = f"test_doc_{i}"
            rag_service.add_document(doc, doc_id)
            logger.info(f"Indexed document {i+1}/{len(documents)}: {doc.metadata.get('title', doc_id)}")
        
        logger.info(f"Successfully indexed {len(documents)} test documents in collection '{collection_name}'")
        
        # Проверяем индексацию простым поиском
        test_query = "Что такое RAG?"
        logger.info(f"Testing search with query: '{test_query}'")
        results = rag_service.search(test_query)
        logger.info(f"Search returned {len(results)} results")
        
        if results:
            logger.info("Search test successful!")
            logger.info("First result snippet: " + results[0][0].text[:100] + "...")
        else:
            logger.warning("Search returned no results. Check Qdrant connection and collection.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test data for RAG evaluation")
    parser.add_argument("--collection", type=str, default="test_documents",
                       help="Name of the collection to create in Qdrant")
    parser.add_argument("--no-clear", action="store_true",
                       help="Don't clear existing collection before adding new documents")
    args = parser.parse_args()
    
    prepare_test_data(
        collection_name=args.collection,
        clear_existing=not args.no_clear
    ) 