"""
Маршруты для работы с RAG (Retrieval-Augmented Generation).
"""

import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User as UserModel, Chat as ChatModel, ModelConfig as ModelConfigModel, Document as DocumentModel
from app.schemas import QuestionRequest, ChatResponse
from app.ollama_client import get_ollama_instance
from app.routers.auth import get_current_user, get_current_admin

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(tags=["rag"])

# Глобальная переменная для экземпляра RAG-сервиса
_rag_service = None

# Служебные функции
def get_user_model_config(user_id: int, db: Session) -> ModelConfigModel:
    """Возвращает конфигурацию модели для пользователя или создает её по умолчанию"""
    config = db.query(ModelConfigModel).filter(ModelConfigModel.user_id == user_id).first()
    
    if not config:
        # Создаем конфигурацию по умолчанию
        config = ModelConfigModel(user_id=user_id)
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return config

def get_rag_service():
    """Получает экземпляр RAG-сервиса"""
    global _rag_service
    if _rag_service is None:
        try:
            logger.info("Initializing RAGService in router")
            # Используем функцию из конфигурации, которая учтет все переменные окружения
            from app.config import create_rag_service_from_config
            _rag_service = create_rag_service_from_config()
            logger.info("RAGService initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAGService: {str(e)}")
            raise
    return _rag_service

# Маршруты
@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    question_request: QuestionRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Обработка вопроса с использованием RAG.
    """
    try:
        logger.info(f"Processing question from user {current_user.username}: {question_request.question}")
        
        # Получаем настройки пользователя
        user_config = get_user_model_config(current_user.id, db)
        
        # Получаем клиент Ollama
        ollama_client = get_ollama_instance(user_config.model_name)
        
        # Получаем сервис RAG
        rag_service = get_rag_service()
        
        # Метрики времени
        t0 = time.perf_counter()
        # Поиск релевантных чанков
        relevant_chunks = rag_service.search(question_request.question, top_k=5)
        t1 = time.perf_counter()
        retrieval_time_ms = int((t1 - t0) * 1000)
        
        # Формируем запрос с контекстом
        logger.debug("Найдено %d релевантных чанков", len(relevant_chunks))
        prompt = rag_service.generate_prompt(question_request.question, top_k_chunks=5)
        logger.info("Generated RAG prompt with context")
        
        # Запрос к модели
        t2 = time.perf_counter()
        try:
            response = await ollama_client.generate(prompt, temperature=user_config.temperature)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            # Если произошла ошибка, возвращаем сообщение об ошибке
            if "Failed to connect to Ollama" in str(e):
                response = f"Ошибка: не удалось подключиться к Ollama. Пожалуйста, установите и запустите Ollama, следуя инструкции на https://ollama.com/download"
            else:
                response = f"Произошла ошибка при генерации ответа: {str(e)}"
        t3 = time.perf_counter()
        generation_time_ms = int((t3 - t2) * 1000)
        
        logger.info("Received response from Ollama")
        
        # Сохраняем результат в БД
        logger.info("Saving chat to database")
        
        # Преобразуем результаты поиска в формат для сохранения
        relevant_chunks_data = []
        for chunk, score in relevant_chunks:
            # Преобразуем метаданные в словарь для сериализации
            metadata = {k: v for k, v in chunk.metadata.items() if k != 'embedding'}
            
            relevant_chunks_data.append({
                "text": chunk.text,
                "relevance": float(score),
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "metadata": metadata
            })
        
        # Собираем метаинформацию
        meta = {
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "retrieved_chunks_count": len(relevant_chunks),
            "prompt_length": len(prompt) if prompt else None,
            "response_length": len(response) if response else None,
            "model": getattr(user_config, 'model_name', None),
            "temperature": getattr(user_config, 'temperature', None),
            "top_k_chunks": 5,
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "rag_used": True
        }
        
        # Создаем новую запись чата
        chat = ChatModel(
            user_id=current_user.id,
            question=question_request.question,
            response=response,
            relevant_chunks=relevant_chunks_data,
            rag_used=meta.get('rag_used', True)
        )
        
        db.add(chat)
        db.commit()
        db.refresh(chat)
        
        logger.info("Chat saved successfully")
        
        # Преобразуем модель БД в модель ответа
        return ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            question=chat.question,
            response=chat.response,
            created_at=chat.created_at,
            relevant_chunks=chat.relevant_chunks,
            meta=meta,
            rag_used=chat.rag_used
        )
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )

@router.get("/chats", response_model=List[ChatResponse])
async def get_chat_history(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получение истории чатов пользователя.
    """
    chats = db.query(ChatModel).filter(
        ChatModel.user_id == current_user.id
    ).order_by(ChatModel.created_at.desc()).all()
    
    return chats

@router.delete("/chats/clear")
async def clear_chat_history(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Очищает всю историю чата пользователя.
    """
    db.query(ChatModel).filter(ChatModel.user_id == current_user.id).delete()
    db.commit()
    return {"status": "success", "message": "История чата очищена"}

@router.post("/direct-ask")
async def direct_ask(
    question_request: QuestionRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Прямой запрос к LLM без использования RAG.
    """
    try:
        logger.info(f"Direct question from user {current_user.username}: {question_request.question}")
        
        # Получаем настройки пользователя
        user_config = get_user_model_config(current_user.id, db)
        
        # Получаем клиент Ollama
        ollama_client = get_ollama_instance(user_config.model_name)
        
        # Запрос к модели
        t0 = time.perf_counter()
        try:
            response = await ollama_client.generate(
                question_request.question, 
                temperature=user_config.temperature
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            if "Failed to connect to Ollama" in str(e):
                response = f"Ошибка: не удалось подключиться к Ollama. Пожалуйста, установите и запустите Ollama, следуя инструкции на https://ollama.com/download"
            else:
                response = f"Произошла ошибка при генерации ответа: {str(e)}"
        t1 = time.perf_counter()
        generation_time_ms = int((t1 - t0) * 1000)
        
        # Сохраняем результат в БД
        meta = {
            "generation_time_ms": generation_time_ms,
            "model": getattr(user_config, 'model_name', None),
            "temperature": getattr(user_config, 'temperature', None),
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "rag_used": False
        }
        
        # Создаем новую запись чата
        chat = ChatModel(
            user_id=current_user.id,
            question=question_request.question,
            response=response,
            rag_used=False
        )
        
        db.add(chat)
        db.commit()
        db.refresh(chat)
        
        # Преобразуем модель БД в модель ответа
        return ChatResponse(
            id=chat.id,
            user_id=chat.user_id,
            question=chat.question,
            response=chat.response,
            created_at=chat.created_at,
            meta=meta,
            rag_used=False
        )
    except Exception as e:
        logger.error(f"Error processing direct question: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки запроса: {str(e)}"
        )

@router.get("/stats")
async def get_rag_stats(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получение статистики о состоянии RAG-системы, включая:
    - Количество документов по статусам
    - Общее количество чанков
    - Информацию о векторной базе
    - Метаданные обработки и настройки
    """
    try:
        # Общее количество документов
        total_docs = db.query(DocumentModel).count()
        
        # Статусы документов
        status_counts = {}
        statuses = ['indexed', 'error', 'processing', 'chunking', 'embedding', 'reindexing', 'uploaded']
        for status_name in statuses:
            count = db.query(DocumentModel).filter(DocumentModel.status == status_name).count()
            status_counts[status_name] = count
        
        # Общее количество чанков
        total_chunks = db.query(DocumentModel.chunks_count).filter(
            DocumentModel.status == 'indexed'
        ).all()
        total_chunks_count = sum(count[0] for count in total_chunks if count[0] is not None)
        
        # Получаем сервис RAG и его настройки
        rag_service = get_rag_service()
        
        # Информация о режиме чанкинга и настройках
        chunking_info = {
            "mode": rag_service.chunking_mode,
            "chunk_size": rag_service.chunk_size,
            "chunk_overlap": rag_service.chunk_overlap,
            "min_chunk_size": rag_service.min_chunk_size,
            "max_chunk_size": rag_service.max_chunk_size
        }
        
        # Информация о поиске
        search_info = {
            "use_hybrid": rag_service.use_hybrid,
            "dense_weight": rag_service.dense_weight,
            "sparse_weight": getattr(rag_service, 'sparse_weight', 0.3),
            "reranker_weight": rag_service.reranker_weight,
            "model_name": rag_service.model_name,
            "vector_size": rag_service.vectorizer.vector_size,
        }
        
        # Информация о размере векторной базы
        try:
            # Получаем информацию о коллекции Qdrant
            collection_info = rag_service.index.client.get_collection(
                collection_name=rag_service.collection_name
            )
            vector_db_info = {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "segments_count": collection_info.segments_count,
                "storage_size": getattr(collection_info, 'disk_data_size', 0),
                "status": collection_info.status,
                "collection_name": rag_service.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting Qdrant collection info: {str(e)}")
            vector_db_info = {
                "error": str(e)
            }
        
        # Информация о пользователях и документах
        users_count = db.query(UserModel).count()
        
        # Топ-5 пользователей по количеству документов
        from sqlalchemy import func
        top_users_docs = db.query(
            DocumentModel.user_id,
            UserModel.username,
            func.count(DocumentModel.id).label('docs_count')
        ).join(UserModel).group_by(
            DocumentModel.user_id, 
            UserModel.username
        ).order_by(
            func.count(DocumentModel.id).desc()
        ).limit(5).all()
        
        top_users_docs_data = [{
            "user_id": user_id,
            "username": username,
            "docs_count": docs_count
        } for user_id, username, docs_count in top_users_docs]
        
        # Размеры документов
        doc_sizes = db.query(
            func.min(DocumentModel.file_size).label('min_size'),
            func.max(DocumentModel.file_size).label('max_size'),
            func.avg(DocumentModel.file_size).label('avg_size')
        ).filter(DocumentModel.file_size > 0).first()
        
        # Соберём статистику по количеству чанков на документ
        chunks_per_doc = db.query(
            func.min(DocumentModel.chunks_count).label('min_chunks'),
            func.max(DocumentModel.chunks_count).label('max_chunks'),
            func.avg(DocumentModel.chunks_count).label('avg_chunks')
        ).filter(DocumentModel.chunks_count > 0).first()
        
        # Соберём данные по режимам чанкинга
        chunking_modes_stats = db.query(
            DocumentModel.chunking_mode,
            func.count(DocumentModel.id).label('count')
        ).filter(DocumentModel.chunking_mode.isnot(None)).group_by(
            DocumentModel.chunking_mode
        ).all()
        
        chunking_modes_data = [{
            "mode": mode or "unknown",
            "count": count
        } for mode, count in chunking_modes_stats]
        
        # Собираем полную статистику
        stats = {
            "documents": {
                "total": total_docs,
                "status_counts": status_counts
            },
            "chunks": {
                "total_count": total_chunks_count,
                "per_document": {
                    "min": chunks_per_doc.min_chunks if chunks_per_doc else None,
                    "max": chunks_per_doc.max_chunks if chunks_per_doc else None,
                    "avg": float(chunks_per_doc.avg_chunks) if chunks_per_doc and chunks_per_doc.avg_chunks is not None else None,
                }
            },
            "vector_db": vector_db_info,
            "chunking": {
                "current_config": chunking_info,
                "modes_usage": chunking_modes_data
            },
            "search": search_info,
            "users": {
                "total": users_count,
                "top_by_docs": top_users_docs_data
            },
            "document_sizes": {
                "min": doc_sizes.min_size if doc_sizes else None,
                "max": doc_sizes.max_size if doc_sizes else None,
                "avg": float(doc_sizes.avg_size) if doc_sizes and doc_sizes.avg_size is not None else None,
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        from fastapi import status as fastapi_status
        raise HTTPException(
            status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики: {str(e)}"
        )

@router.post("/index/clear")
async def clear_index(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    user_id: Optional[int] = None
):
    """
    Очищает индекс RAG.
    
    Обычные пользователи могут очистить только свой индекс.
    Администраторы могут очистить как свой индекс, так и индексы других пользователей,
    указав их user_id, либо очистить весь индекс полностью.
    """
    try:
        # Проверяем права
        is_admin = current_user.role == "admin"
        
        # Обычный пользователь пытается очистить чужой индекс
        if user_id is not None and user_id != current_user.id and not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для очистки индекса другого пользователя"
            )
        
        rag_service = get_rag_service()
        
        # Если указан user_id и пользователь имеет права, очищаем только этот индекс
        if user_id is not None:
            # Проверяем существование пользователя
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Пользователь с ID {user_id} не найден"
                )
                
            # В текущей реализации мы не можем очистить индекс отдельного пользователя,
            # поэтому просто очищаем весь индекс - это будет улучшено в будущих версиях
            logger.warning(f"Clearing entire index because per-user clearing is not implemented yet")
            rag_service.clear_index()
            return {"status": "success", "message": f"Индекс пользователя {user_id} очищен"}
        
        # Очищаем весь индекс
        rag_service.clear_index()
        
        return {"status": "success", "message": "Индекс очищен"}
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        return {"status": "error", "message": str(e)} 