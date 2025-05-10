"""
Маршруты для работы с RAG (Retrieval-Augmented Generation).
"""

import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User as UserModel, Chat as ChatModel, ModelConfig as ModelConfigModel
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