#!/usr/bin/env python
"""
Скрипт для перестроения векторного индекса из документов, уже загруженных в базу данных.
"""

import os
import logging
import asyncio
import shutil
from pathlib import Path
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import User, Document
from app.rag import RAGService, Document as RAGDocument

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Директория индексов
INDEX_DIR = os.getenv("INDEX_DIR", "indexes")

async def rebuild_index():
    """Перестраивает индекс из существующих документов в БД"""
    # Создание сессии БД
    db = SessionLocal()
    
    try:
        # Получаем все документы из базы
        documents = db.query(Document).all()
        logger.info(f"Найдено {len(documents)} документов в БД для индексации")
        
        if not documents:
            logger.error("Документов в БД нет, индексация невозможна")
            return
        
        # Очищаем директорию индексов
        index_dir_path = Path(INDEX_DIR)
        if index_dir_path.exists():
            # Сначала сохраняем список файлов в директории
            files_before = list(index_dir_path.glob("*"))
            
            # Создаем временный RAGService, чтобы получить доступ к методу clear индекса
            temp_rag = RAGService(index_name="default")
            temp_rag.index.clear()
            logger.info("Текущий индекс очищен")
            
        # Создаем новый RAGService (с пустым индексом)
        rag_service = RAGService(index_name="default")
        
        # Индексируем каждый документ
        total_chunks = 0
        for doc in documents:
            # Создаем RAG документ
            rag_doc = RAGDocument(
                content=doc.content,
                metadata={
                    'title': doc.title,
                    'source': doc.source,
                    'doc_id': str(doc.id)
                }
            )
            
            # Индексация документа
            doc_id = f"doc_{doc.id}"
            try:
                logger.info(f"Индексация документа: {doc.title} (ID: {doc.id})")
                chunk_ids = rag_service.add_document(rag_doc, doc_id)
                logger.info(f"Документ '{doc.title}' успешно индексирован, создано {len(chunk_ids) if chunk_ids else 0} чанков")
                total_chunks += len(chunk_ids) if chunk_ids else 0
            except Exception as e:
                logger.error(f"Ошибка при индексации документа '{doc.title}': {str(e)}")
        
        # Проверяем, что индекс содержит данные
        index_check = RAGService()
        test_query = "RAG"
        results = index_check.search(test_query, top_k=1)
        if results:
            logger.info(f"Индекс успешно создан и содержит данные. Пример найденного чанка: {results[0][0].text[:50]}...")
        else:
            logger.warning("Индекс создан, но поиск не вернул результатов")
            
    except Exception as e:
        logger.error(f"Ошибка при перестроении индекса: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(rebuild_index()) 