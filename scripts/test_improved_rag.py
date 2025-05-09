#!/usr/bin/env python
"""
Демонстрационный скрипт для тестирования улучшенной RAG-системы.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from app.rag import RAGService, Document
from app.ollama_client import OllamaLLM

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Установка значений по умолчанию из переменных окружения
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CROSS_ENCODER = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_LLM = os.getenv("LLM_MODEL", "mistral")

def load_test_documents() -> List[Document]:
    """
    Загружает тестовые документы из директории docs/.
    
    Returns:
        Список документов
    """
    docs_dir = Path("docs")
    documents = []
    
    # Проверяем, что директория существует
    if not docs_dir.exists():
        logger.warning(f"Directory {docs_dir} does not exist!")
        return documents
    
    # Загружаем все .txt файлы
    for file_path in docs_dir.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            metadata = {
                "source": file_path.name,
                "file_path": str(file_path),
                "created_at": time.time()
            }
            
            document = Document(content=content, metadata=metadata)
            documents.append(document)
            
            logger.info(f"Loaded document: {file_path.name} ({len(content)} characters)")
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
    
    return documents

def setup_rag_service() -> RAGService:
    """
    Создает и настраивает улучшенную RAG-систему.
    
    Returns:
        Экземпляр RAGService
    """
    # Создаем RAG-сервис с улучшениями
    service = RAGService(
        index_name="improved_rag_test",
        model_name=DEFAULT_MODEL,
        chunk_size=300,
        chunk_overlap=100,
        use_hybrid_search=True,
        use_reranker=True,
        dense_weight=0.7,
        reranker_weight=0.5,
        use_adaptive_k=True,
        cross_encoder_model=DEFAULT_CROSS_ENCODER,
        language="russian"
    )
    
    logger.info(f"Created RAG service with embedding model: {DEFAULT_MODEL}")
    logger.info(f"Using hybrid search with dense_weight: 0.7")
    if service.use_hybrid_search and service.hybrid_retriever:
        logger.info("Hybrid search enabled ✅")
        logger.info(f"Using adaptive top_k: {service.use_adaptive_k}")
        
        if service.use_reranker and service.hybrid_retriever.reranker and service.hybrid_retriever.reranker.is_available():
            logger.info(f"CrossEncoder reranker enabled ✅: {DEFAULT_CROSS_ENCODER}")
        else:
            logger.warning("CrossEncoder reranker not available ❌")
    else:
        logger.warning("Hybrid search not available ❌")
    
    return service

def index_documents(service: RAGService, documents: List[Document]) -> None:
    """
    Индексирует документы в RAG-системе.
    
    Args:
        service: Экземпляр RAG-сервиса
        documents: Список документов для индексации
    """
    start_time = time.time()
    
    total_chunks = 0
    for i, doc in enumerate(documents):
        doc_id = f"doc_{i}"
        chunks = service.add_document(doc, doc_id)
        total_chunks += len(chunks)
        
        logger.info(f"Indexed document {doc_id} with {len(chunks)} chunks")
    
    duration = time.time() - start_time
    logger.info(f"Indexed {len(documents)} documents with {total_chunks} total chunks in {duration:.2f} seconds")

def test_search(service: RAGService, queries: List[str]) -> None:
    """
    Тестирует поиск в RAG-системе.
    
    Args:
        service: Экземпляр RAG-сервиса
        queries: Список запросов для тестирования
    """
    for i, query in enumerate(queries):
        logger.info(f"\n--- Query {i+1}: {query} ---")
        
        start_time = time.time()
        results = service.search(query, top_k=5)
        duration = time.time() - start_time
        
        logger.info(f"Search completed in {duration:.3f} seconds, found {len(results)} results")
        
        # Выводим результаты
        for j, (chunk, score) in enumerate(results):
            # Обрезаем текст для более компактного вывода
            text = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            
            # Выводим метаданные
            source = chunk.metadata.get("source", "unknown")
            content_type = chunk.metadata.get("content_type", "text")
            
            logger.info(f"{j+1}. Score: {score:.4f} | Source: {source} | Type: {content_type}")
            logger.info(f"   Text: {text}")
            logger.info("-" * 80)

def test_generation(service: RAGService, llm: OllamaLLM, queries: List[str]) -> None:
    """
    Тестирует генерацию ответов с использованием RAG.
    
    Args:
        service: Экземпляр RAG-сервиса
        llm: Экземпляр LLM для генерации
        queries: Список запросов для тестирования
    """
    for i, query in enumerate(queries):
        logger.info(f"\n=== Query {i+1}: {query} ===")
        
        # Получаем промпт
        start_time = time.time()
        prompt = service.generate_prompt(query, top_k_chunks=5)
        prompt_time = time.time() - start_time
        
        logger.info(f"Prompt generated in {prompt_time:.3f} seconds")
        
        # Генерируем ответ
        start_time = time.time()
        response = llm.generate_sync(prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"Response generated in {generation_time:.3f} seconds")
        logger.info(f"\nОтвет:\n{response}")
        logger.info("=" * 80)

def main():
    """Основная функция программы."""
    logger.info("Starting improved RAG test script")
    
    # Загружаем тестовые документы
    documents = load_test_documents()
    if not documents:
        logger.error("No documents loaded. Please add documents to the docs/ directory.")
        return
    
    # Создаем и настраиваем RAG-сервис
    service = setup_rag_service()
    
    # Индексируем документы
    index_documents(service, documents)
    
    # Тестовые запросы
    test_queries = [
        "Что такое RAG?",
        "Преимущества гибридного поиска",
        "Как улучшить качество извлечения документов?",
        "Расскажи про методы chunking"
    ]
    
    # Тестируем поиск
    test_search(service, test_queries)
    
    # Тестируем генерацию с LLM
    try:
        # Инициализируем LLM
        llm = OllamaLLM(model_name=DEFAULT_LLM)
        llm.ensure_model_loaded_sync()
        
        # Тестируем генерацию
        test_generation(service, llm, test_queries)
    except Exception as e:
        logger.error(f"Failed to initialize or use LLM: {str(e)}")
        logger.info("Skipping generation tests.")

if __name__ == "__main__":
    main() 