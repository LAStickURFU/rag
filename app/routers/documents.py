"""
Маршруты для работы с документами - загрузка, индексация, получение и удаление.
"""

import os
import logging
import uuid
import mimetypes
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User as UserModel, Document as DocumentModel
from app.routers.auth import get_current_user, get_current_admin

# Импортируем функцию для извлечения текста
from app.utils.parser.document_parser import extract_text

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(prefix="/documents", tags=["documents"])

# Получение RAG-сервиса из модуля routers.rag
def get_rag_service():
    """Получает экземпляр RAG-сервиса"""
    from app.routers.rag import get_rag_service as rag_get_rag_service
    return rag_get_rag_service()

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    titles: List[str] = Form(...),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Загружает документы в систему и индексирует их для RAG.
    """
    if len(files) != len(titles):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Количество файлов и заголовков должно совпадать"
        )
    
    results = []
    for i, file in enumerate(files):
        # Создаем уникальный ID для документа сразу
        doc_uuid = str(uuid.uuid4())
        safe_filename = ""
        
        try:
            title = titles[i]
            logger.info(f"Обработка файла: {file.filename}, название: {title}")
            
            # Читаем содержимое файла
            try:
                file_content = await file.read()
                
                # Проверяем размер файла
                if len(file_content) == 0:
                    logger.error(f"Файл {file.filename} пустой")
                    raise ValueError(f"Файл {file.filename} не содержит данных")
            except Exception as e:
                logger.error(f"Ошибка чтения файла {file.filename}: {str(e)}")
                raise ValueError(f"Ошибка чтения файла: {str(e)}")
            
            # Определяем MIME-тип файла
            content_type, _ = mimetypes.guess_type(file.filename)
            
            # Если тип не определен, используем расширение
            if not content_type:
                ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
                content_type = f"application/{ext}" if ext else "application/octet-stream"
            
            # Убедимся, что имя файла не пустое
            safe_filename = file.filename or f"{title}.txt"
            logger.info(f"Тип контента: {content_type}, имя файла: {safe_filename}")
            
            # Создаем запись в БД до извлечения текста, чтобы сохранить информацию даже при ошибке
            document = DocumentModel(
                title=title,
                content="",  # Заполним после извлечения текста
                source="manual_upload",
                user_id=current_user.id,
                file_name=safe_filename,
                file_size=len(file_content),
                uuid=doc_uuid,
                status="processing"  # Статус "в обработке"
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Извлекаем текст из файла
            text_content = ""
            try:
                logger.info(f"Начинаем извлечение текста из {file.filename}")
                text_content = extract_text(file, file_content)
                logger.info(f"Текст успешно извлечен, длина: {len(text_content)} символов")
                
                # После успешного извлечения текста обновляем запись
                document.content = text_content
                document.status = "indexing"  # Меняем статус на "индексируется"
                db.commit()
                
            except Exception as e:
                logger.error(f"Ошибка извлечения текста из {file.filename}: {str(e)}")
                document.status = "error"
                document.error_message = str(e)
                db.commit()
                
                results.append({
                    "id": document.id,
                    "title": document.title,
                    "filename": document.file_name,
                    "error": str(e),
                    "status": "error"
                })
                continue  # Переходим к следующему файлу

            # Добавляем документ в RAG-индекс
            try:
                rag_service = get_rag_service()
                
                # Получаем текущие настройки RAG-сервиса для метаданных
                current_settings = {
                    "chunking_mode": rag_service.chunking_mode,
                    "chunk_size": rag_service.chunk_size,
                    "chunk_overlap": rag_service.chunk_overlap,
                    "embedding_model": rag_service.model_name,
                    "use_hybrid_search": rag_service.use_hybrid,
                    "reranker_weight": rag_service.reranker_weight,
                    "dense_weight": rag_service.dense_weight,
                    "sparse_weight": getattr(rag_service, 'sparse_weight', 0.3),
                }
                
                # Создаем документ для RAG
                from app.rag import Document as RagDocument
                rag_doc = RagDocument(
                    text=text_content,
                    metadata={
                        "title": title,
                        "source": "manual_upload",
                        "filename": safe_filename,
                        "filetype": content_type,
                        "user_id": current_user.id,
                        "doc_id": doc_uuid,
                        "chunking_mode": current_settings["chunking_mode"]
                    }
                )
                
                # Индексируем документ
                chunks = rag_service.add_document(rag_doc, doc_id=doc_uuid)
                
                # Обновляем статус и количество чанков
                document.status = "indexed"
                document.chunks_count = len(chunks) if chunks else 0
                
                # Сохраняем метаинформацию о процессе обработки
                document.chunking_mode = current_settings["chunking_mode"]
                document.chunk_size = current_settings["chunk_size"]
                document.chunk_overlap = current_settings["chunk_overlap"]
                document.embedding_model = current_settings["embedding_model"]
                
                # Сохраняем дополнительные параметры в JSON
                document.processing_params = {
                    "use_hybrid_search": current_settings["use_hybrid_search"],
                    "reranker_weight": current_settings["reranker_weight"],
                    "dense_weight": current_settings["dense_weight"],
                    "sparse_weight": current_settings["sparse_weight"],
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "content_type": content_type
                }
                
                # Логируем информацию о загрузке
                logger.info(f"Документ {document.id} ({document.title}) успешно загружен: режим={document.chunking_mode}, чанков={document.chunks_count}")
                
                db.commit()
                
                # Добавляем результат
                results.append({
                    "id": document.id,
                    "title": document.title,
                    "filename": document.file_name,
                    "size": document.file_size,
                    "chunks_count": document.chunks_count,
                    "status": document.status
                })
                
            except Exception as e:
                logger.error(f"Ошибка индексирования документа {file.filename}: {str(e)}")
                document.status = "error"
                document.error_message = f"Ошибка индексирования: {str(e)}"
                db.commit()
                
                results.append({
                    "id": document.id,
                    "title": document.title,
                    "filename": document.file_name,
                    "error": str(e),
                    "status": "error"
                })
            
        except Exception as e:
            # Эта часть обрабатывает ошибки, возникшие до создания документа в БД
            logger.error(f"Общая ошибка обработки документа {getattr(file, 'filename', 'unknown')}: {str(e)}")
            
            try:
                # Если запись в БД уже существует, обновляем её
                if 'document' in locals() and document:
                    document.status = "error"
                    document.error_message = str(e)
                    db.commit()
                    
                    results.append({
                        "id": document.id,
                        "title": document.title,
                        "filename": document.file_name,
                        "error": str(e),
                        "status": "error"
                    })
                else:
                    # Создаем новую запись об ошибке
                    error_doc = DocumentModel(
                        title=titles[i] if i < len(titles) else getattr(file, 'filename', 'unknown'),
                        content="",
                        source="manual_upload",
                        user_id=current_user.id,
                        file_name=safe_filename or getattr(file, 'filename', 'unknown'),
                        file_size=0,
                        uuid=doc_uuid,
                        status="error",
                        error_message=str(e)
                    )
                    db.add(error_doc)
                    db.commit()
                    
                    results.append({
                        "id": error_doc.id,
                        "title": error_doc.title,
                        "filename": error_doc.file_name,
                        "error": str(e),
                        "status": "error"
                    })
            except Exception as inner_e:
                logger.error(f"Ошибка сохранения информации об ошибке: {str(inner_e)}")
                results.append({
                    "filename": getattr(file, 'filename', 'unknown'),
                    "error": f"{str(e)} (а также ошибка сохранения: {str(inner_e)})",
                    "status": "error"
                })
    
    return {"status": "success", "documents": results}

@router.get("")
async def get_documents(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    all_users: bool = False
):
    """
    Получает список документов пользователя.
    Если пользователь администратор и указан параметр all_users=True,
    возвращает документы всех пользователей.
    """
    # Проверяем права на получение всех документов
    if all_users and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для просмотра документов всех пользователей"
        )
    
    # Формируем запрос
    query = db.query(DocumentModel)
    if not all_users or current_user.role != "admin":
        # Для обычных пользователей или если не запрошены все документы
        query = query.filter(DocumentModel.user_id == current_user.id)
    
    documents = query.order_by(DocumentModel.created_at.desc()).all()
    
    # Преобразуем в словари для ответа
    result = []
    for doc in documents:
        # Получаем данные о пользователе, загрузившем документ
        uploader = db.query(UserModel).filter(UserModel.id == doc.user_id).first()
        uploader_name = uploader.username if uploader else "Unknown"
        
        result.append({
            "id": doc.id,
            "title": doc.title,
            "filename": doc.file_name,
            "size": doc.file_size,
            "chunks_count": doc.chunks_count,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "uuid": doc.uuid,
            "user_id": doc.user_id,
            "uploader": uploader_name,
            "chunking_mode": doc.chunking_mode,
            "embedding_model": doc.embedding_model,
            "processing_summary": {
                "chunk_size": doc.chunk_size,
                "chunk_overlap": doc.chunk_overlap,
                "has_additional_params": bool(doc.processing_params)
            }
        })
    
    return result

@router.post("/reindex")
async def reindex_documents(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    all_users: bool = False
):
    """
    Переиндексирует документы пользователя.
    Если пользователь администратор и указан параметр all_users=True,
    переиндексирует документы всех пользователей.
    """
    try:
        # Проверяем права на переиндексацию всех документов
        if all_users and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Недостаточно прав для переиндексации документов всех пользователей"
            )
        
        # Формируем запрос
        query = db.query(DocumentModel)
        if not all_users or current_user.role != "admin":
            # Для обычных пользователей или если не запрошены все документы
            query = query.filter(DocumentModel.user_id == current_user.id)
        
        documents = query.all()
        
        if not documents:
            return {"status": "success", "message": "Нет документов для индексации"}
        
        # Очищаем индекс
        rag_service = get_rag_service()
        rag_service.clear_index()
        
        # Индексируем каждый документ
        reindexed_count = 0
        total_chunks = 0
        for doc in documents:
            if not doc.content:
                continue
                
            # Меняем статус на "индексируется"
            doc.status = "indexing"
            db.commit()
            
            try:
                # Создаем документ для RAG
                from app.rag import Document as RagDocument
                rag_doc = RagDocument(
                    text=doc.content,
                    metadata={
                        "title": doc.title,
                        "source": doc.source,
                        "filename": doc.file_name,
                        "user_id": doc.user_id,
                        "doc_id": doc.uuid or str(uuid.uuid4())
                    }
                )
                
                # Если у документа нет UUID, создаем его
                if not doc.uuid:
                    doc.uuid = rag_doc.metadata["doc_id"]
                
                # Получаем текущие настройки RAG-сервиса для метаинформации
                current_settings = {
                    "chunking_mode": rag_service.chunking_mode,
                    "chunk_size": rag_service.chunk_size,
                    "chunk_overlap": rag_service.chunk_overlap,
                    "embedding_model": rag_service.model_name,
                    "use_hybrid_search": rag_service.use_hybrid,
                    "reranker_weight": rag_service.reranker_weight,
                    "dense_weight": rag_service.dense_weight,
                    "sparse_weight": getattr(rag_service, 'sparse_weight', 0.3),
                }
                
                # Индексируем документ
                chunks = rag_service.add_document(rag_doc, doc_id=doc.uuid)
                
                # Обновляем статус и все метаданные документа согласно текущим настройкам
                doc.status = "indexed"
                chunks_count = len(chunks) if chunks else 0
                doc.chunks_count = chunks_count
                
                # Обновляем всю метаинформацию о документе
                doc.chunking_mode = current_settings["chunking_mode"]
                doc.chunk_size = current_settings["chunk_size"]
                doc.chunk_overlap = current_settings["chunk_overlap"]
                doc.embedding_model = current_settings["embedding_model"]
                
                # Сохраняем дополнительные параметры в JSON
                doc.processing_params = {
                    "use_hybrid_search": current_settings["use_hybrid_search"],
                    "reranker_weight": current_settings["reranker_weight"],
                    "dense_weight": current_settings["dense_weight"],
                    "sparse_weight": current_settings["sparse_weight"],
                    "updated_at": datetime.utcnow().isoformat(),
                }
                
                # Добавляем комментарий в лог о выполняемом обновлении метаданных
                logger.info(f"Updating metadata for document {doc.id}: mode={doc.chunking_mode}, chunks={chunks_count}, embedding_model={doc.embedding_model}")
                
                total_chunks += chunks_count
                reindexed_count += 1
                
            except Exception as e:
                logger.error(f"Error reindexing document {doc.id}: {str(e)}")
                doc.status = "error"
                doc.error_message = str(e)
                
            db.commit()
        
        return {
            "status": "success", 
            "message": f"Переиндексировано {reindexed_count} из {len(documents)} документов",
            "indexed_documents": reindexed_count,
            "total_chunks": total_chunks
        }
        
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при переиндексации: {str(e)}"
        )

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Удаляет документ и его индексы.
    Администраторы могут удалять любые документы,
    обычные пользователи - только свои.
    """
    # Проверяем существование документа
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Документ не найден"
        )
    
    # Проверяем права доступа
    if document.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Вы можете удалять только собственные документы"
        )
    
    try:
        # Удаляем документ из индекса, если у него есть UUID
        if document.uuid:
            rag_service = get_rag_service()
            rag_service.delete_document(document.uuid)
        
        # Удаляем документ из БД
        db.delete(document)
        db.commit()
        
        return {"status": "success", "message": "Документ успешно удален"}
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении документа: {str(e)}"
        )

@router.delete("/all/clear")
async def delete_all_documents(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Удаляет все документы из базы данных.
    Доступно только администраторам.
    """
    try:
        # Получаем все документы
        documents = db.query(DocumentModel).all()
        
        if not documents:
            return {"status": "success", "message": "Нет документов для удаления"}
        
        # Очищаем индекс
        rag_service = get_rag_service()
        rag_service.clear_index()
        
        # Удаляем все документы
        count = db.query(DocumentModel).delete()
        db.commit()
        
        return {
            "status": "success", 
            "message": f"Удалено {count} документов"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении документов: {str(e)}"
        )

@router.get("/{document_id}")
async def get_document_detail(
    document_id: int, 
    current_user: UserModel = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """
    Получает детальную информацию о документе.
    """
    document = db.query(DocumentModel).filter(
        DocumentModel.id == document_id,
        DocumentModel.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Документ не найден или доступ запрещен"
        )
    
    # Возвращаем детали документа, но без полного содержимого для экономии трафика
    content_preview = document.content[:500] + "..." if document.content and len(document.content) > 500 else document.content
    
    # Рассчитываем и добавляем дополнительные метаданные
    content_length = len(document.content) if document.content else 0
    
    # Информация о чанкинге
    chunking_info = {
        "mode": document.chunking_mode or "character",
        "chunk_size": document.chunk_size,
        "chunk_overlap": document.chunk_overlap,
        "embedding_model": document.embedding_model,
    }
    
    # Дополнительные параметры из JSON
    processing_info = document.processing_params or {}
    
    return {
        "id": document.id,
        "title": document.title,
        "filename": document.file_name or "-",
        "size": document.file_size or 0,
        "chunks_count": document.chunks_count or 0,
        "status": document.status or "unknown",
        "source": document.source or "-",
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document.updated_at else None,
        "uuid": document.uuid,
        "content_preview": content_preview,
        "error_message": document.error_message,
        "content_length": content_length,  # Длина текста в символах
        "chunking_mode": document.chunking_mode,
        "chunking_info": chunking_info,
        "processing_info": processing_info
    } 