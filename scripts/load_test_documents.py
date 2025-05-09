#!/usr/bin/env python
"""
Скрипт для загрузки тестовых документов из директории docs/ в RAG-систему.
Этот скрипт создает тестового пользователя (если он не существует) и загружает
все markdown-файлы из директории docs/ в базу данных и FAISS-индекс.
"""

import os
import glob
import logging
import asyncio
from pathlib import Path
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Перезагружаем переменные окружения
load_dotenv(override=True)

from app.database import SessionLocal
from app.models import User, Document
from app.rag import RAGService, Document as RAGDocument
from app.ollama_client import get_ollama_instance
from passlib.context import CryptContext
from app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, USE_HYBRID_SEARCH, USE_RERANKER, CROSS_ENCODER_MODEL, LANGUAGE, SPACY_MODEL

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Инициализация хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def create_test_user(db: Session):
    """Создает тестового пользователя, если он не существует"""
    test_username = "test_user"
    test_password = "test_password"
    
    existing_user = db.query(User).filter(User.username == test_username).first()
    if existing_user:
        logger.info(f"Тестовый пользователь '{test_username}' уже существует")
        return existing_user
    
    hashed_password = get_password_hash(test_password)
    new_user = User(username=test_username, hashed_password=hashed_password)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"Создан тестовый пользователь: {test_username}")
    return new_user

def load_markdown_files(docs_dir):
    """Загружает содержимое всех markdown файлов из указанной директории"""
    markdown_files = glob.glob(os.path.join(docs_dir, "*.md"))
    
    documents = []
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            title = filename.replace('.md', '').replace('_', ' ').title()
            
            documents.append({
                'title': title,
                'content': content,
                'source': 'test_docs',
                'filename': filename
            })
            
            logger.info(f"Загружен документ: {title} ({len(content)} символов)")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path}: {str(e)}")
    
    return documents

async def main():
    """Основная функция для загрузки документов"""
    # Вывод информации о используемых настройках
    logger.info(f"Используется модель эмбеддингов: {EMBEDDING_MODEL}")
    logger.info(f"Используется модель spaCy: {SPACY_MODEL}")
    logger.info(f"Размер фрагмента: {CHUNK_SIZE}, перекрытие: {CHUNK_OVERLAP}")
    logger.info(f"Гибридный поиск: {USE_HYBRID_SEARCH}, переранжирование: {USE_RERANKER}")
    
    # Инициализация RAG сервиса с параметрами из config
    rag_service = RAGService(
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        use_hybrid_search=USE_HYBRID_SEARCH,
        use_reranker=USE_RERANKER,
        cross_encoder_model=CROSS_ENCODER_MODEL,
        language=LANGUAGE,
        spacy_model=SPACY_MODEL
    )
    
    # Проверка наличия Ollama модели
    ollama = get_ollama_instance()
    try:
        await ollama.ensure_model_loaded()
        logger.info("Ollama модель успешно загружена")
    except Exception as e:
        logger.warning(f"Не удалось загрузить Ollama модель: {str(e)}")
    
    # Создание сессии БД
    db = SessionLocal()
    
    try:
        # Создание тестового пользователя
        test_user = create_test_user(db)
        
        # Загрузка документов из директории docs/
        docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
        logger.info(f"Загрузка документов из директории: {docs_dir}")
        markdown_documents = load_markdown_files(docs_dir)
        
        # Добавление документов в БД и индексация
        for doc_info in markdown_documents:
            # Проверка, существует ли уже документ с таким названием
            existing_doc = db.query(Document).filter(
                Document.title == doc_info['title'],
                Document.user_id == test_user.id
            ).first()
            
            if existing_doc:
                logger.info(f"Документ '{doc_info['title']}' уже существует в БД, пропускаем")
                continue
            
            # Создание записи в БД
            new_doc = Document(
                title=doc_info['title'],
                content=doc_info['content'],
                source=doc_info['source'],
                user_id=test_user.id
            )
            
            db.add(new_doc)
            db.commit()
            db.refresh(new_doc)
            
            # Создание RAG документа и добавление в индекс
            rag_doc = RAGDocument(
                content=doc_info['content'],
                metadata={
                    'title': doc_info['title'],
                    'source': doc_info['source'],
                    'filename': doc_info['filename'],
                    'doc_id': str(new_doc.id)
                }
            )
            
            # Индексация документа
            doc_id = f"doc_{new_doc.id}"
            try:
                chunk_ids = rag_service.add_document(rag_doc, doc_id)
                logger.info(f"Документ '{doc_info['title']}' успешно индексирован, создано {len(chunk_ids)} чанков")
            except Exception as e:
                logger.error(f"Ошибка при индексации документа '{doc_info['title']}': {str(e)}")
        
        logger.info("Загрузка тестовых документов завершена")
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении скрипта: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main()) 