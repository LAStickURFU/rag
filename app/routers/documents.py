"""
Маршруты для работы с документами - загрузка, индексация, получение и удаление.
"""

import os
import logging
import uuid
import mimetypes
from typing import List, Optional
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
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

# Создание пула потоков для обработки документов
document_executor = ThreadPoolExecutor(max_workers=2)

# Получение RAG-сервиса из модуля routers.rag
def get_rag_service():
    """Получает экземпляр RAG-сервиса"""
    from app.routers.rag import get_rag_service as rag_get_rag_service
    return rag_get_rag_service()

def process_document_background(document_id, file_content, file_name, user_id, doc_uuid):
    """
    Фоновая функция для обработки документа. Запускается в отдельном потоке.
    """
    logger.info(f"Начало фоновой обработки документа {document_id}, имя файла: {file_name}")
    
    # Создаем новую сессию БД для этого потока
    from app.database import get_db
    db = next(get_db())
    
    try:
        # Получаем документ по ID
        document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if not document:
            logger.error(f"Документ {document_id} не найден в БД")
            return
        
        # Обновляем статус на "обработка"
        document.status = "processing"
        db.commit()
        logger.info(f"Документ {document_id} статус изменен на 'processing'")
        
        # Извлекаем текст из файла
        try:
            logger.info(f"Начинаем извлечение текста из {file_name}")
            
            # Создаём файлоподобный объект из контента
            from io import BytesIO
            file_object = BytesIO(file_content)
            file_object.filename = file_name  # Устанавливаем имя файла
            
            # Импортируем функцию для извлечения текста
            from app.utils.parser.document_parser import extract_text
            
            text_content = extract_text(file_object, file_content)
            logger.info(f"Текст успешно извлечен, длина: {len(text_content)} символов")
            
            # После успешного извлечения текста обновляем запись
            document.content = text_content
            document.status = "chunking"  # Меняем статус на "разбиение на чанки"
            db.commit()
            logger.info(f"Документ {document_id} статус изменен на 'chunking'")
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из {file_name}: {str(e)}", exc_info=True)
            document.status = "error"
            document.error_message = f"Ошибка при извлечении текста: {str(e)}"
            db.commit()
            return
        
        # Добавляем документ в RAG-индекс
        try:
            rag_service = get_rag_service()
            logger.info(f"Получен сервис RAG для документа {document_id}")
                        
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
                    "title": document.title,
                    "source": "manual_upload",
                    "filename": document.file_name,
                    "filetype": document.processing_params.get("content_type") if document.processing_params else "text/plain",
                    "user_id": user_id,
                    "doc_id": doc_uuid,
                    "chunking_mode": current_settings["chunking_mode"]
                }
            )
            
            # Обновляем статус на "создание эмбеддингов"
            document.status = "embedding"
            db.commit()
            logger.info(f"Документ {document_id} статус изменен на 'embedding'")
            
            # Индексируем документ
            logger.info(f"Начинаем индексацию документа {document_id}")
            chunks = rag_service.add_document(rag_doc, doc_id=doc_uuid)
            logger.info(f"Документ {document_id} успешно индексирован, создано {len(chunks) if chunks else 0} фрагментов")
            
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
                "processed_at": datetime.utcnow().isoformat(),
            }
            
            # Логируем информацию о загрузке
            logger.info(f"Документ {document.id} ({document.title}) успешно загружен: режим={document.chunking_mode}, чанков={document.chunks_count}")
            
            db.commit()
            logger.info(f"Документ {document_id} статус изменен на 'indexed'")
            
        except Exception as e:
            logger.error(f"Ошибка индексирования документа {file_name}: {str(e)}", exc_info=True)
            document.status = "error"
            document.error_message = f"Ошибка индексирования: {str(e)}"
            db.commit()
    
    except Exception as e:
        logger.error(f"Общая ошибка в process_document_background для документа {document_id}: {str(e)}", exc_info=True)
    finally:
        db.close()
        logger.info(f"Завершена обработка документа {document_id}")

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    titles: List[str] = Form(...),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None
):
    """
    Загружает документы в систему и запускает их асинхронную обработку.
    Сразу возвращает информацию о созданных документах, пока они еще обрабатываются в фоне.
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
                status="uploaded"  # Начальный статус "загружен"
            )

            # Сохраняем начальную метаинформацию
            document.processing_params = {
                "content_type": content_type,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Запускаем обработку документа в фоновом режиме
            document_executor.submit(
                process_document_background,
                document.id,
                file_content,
                safe_filename,
                current_user.id,
                doc_uuid
            )
            
            # Добавляем информацию о документе в результат
            results.append({
                "id": document.id,
                "title": document.title,
                "filename": document.file_name,
                "size": document.file_size,
                "status": document.status,
                "uuid": document.uuid
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
    
    return {"status": "success", "documents": results, "message": "Документы приняты на обработку"}

@router.get("")
async def get_documents(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    all_users: bool = False,
    page: int = 0,
    page_size: int = 100,
    return_all: bool = False
):
    """
    Получает список документов пользователя с поддержкой пагинации.
    
    - Если пользователь администратор и указан параметр all_users=True, 
      возвращает документы всех пользователей.
    - Поддерживает пагинацию с параметрами page и page_size.
    - Если указан параметр return_all=True, игнорирует пагинацию и возвращает все документы,
      что обеспечивает обратную совместимость со вспомогательными скриптами.
    """
    # Проверяем и корректируем параметры пагинации
    if page < 0:
        page = 0
    
    # Ограничиваем размер страницы разумным значением
    if page_size <= 0:
        page_size = 10
    elif page_size > 500:
        page_size = 500
        
    # Проверяем права на получение всех документов
    if all_users and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для просмотра документов всех пользователей"
        )
    
    # Формируем базовый запрос
    query = db.query(DocumentModel)
    if not all_users or current_user.role != "admin":
        # Для обычных пользователей или если не запрошены все документы
        query = query.filter(DocumentModel.user_id == current_user.id)
    
    # Получаем общее количество документов для метаданных пагинации
    total_docs = query.count()
    
    # Добавляем сортировку
    query = query.order_by(DocumentModel.created_at.desc())
    
    # Применяем пагинацию, если не запрошены все документы
    if not return_all:
        query = query.offset(page * page_size).limit(page_size)
    
    documents = query.all()
    
    # Преобразуем в словари для ответа
    result = []
    for doc in documents:
        # Получаем данные о пользователе, загрузившем документ
        show_owner_info = current_user.role == "admin"  # Для админов всегда показываем владельца
        
        uploader = None
        uploader_name = "Unknown"
        if show_owner_info:
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
            "user_id": doc.user_id if show_owner_info else None,
            "uploader": uploader_name if show_owner_info else None,
            "chunking_mode": doc.chunking_mode,
            "embedding_model": doc.embedding_model,
            "processing_summary": {
                "chunk_size": doc.chunk_size,
                "chunk_overlap": doc.chunk_overlap,
                "has_additional_params": bool(doc.processing_params)
            }
        })
    
    # Для обратной совместимости - если запрашиваются все документы без пагинации
    # возвращаем просто список документов как раньше
    if return_all:
        return result
    
    # В случае использования пагинации возвращаем расширенный ответ с метаданными
    return {
        "items": result,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total_docs,
            "total_pages": (total_docs + page_size - 1) // page_size,
            "has_next": (page + 1) * page_size < total_docs,
            "has_prev": page > 0
        }
    }

@router.post("/reindex")
async def reindex_documents(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
    all_users: bool = False
):
    """
    Запускает асинхронную переиндексацию документов пользователя или всех документов.
    Сразу возвращает статус, а обработка продолжается в фоне.
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
        
        # Устанавливаем статус "reindexing" для всех документов
        for doc in documents:
            doc.status = "reindexing"
        db.commit()

        # Запускаем переиндексацию в фоновом потоке
        document_executor.submit(
            reindex_documents_background,
            documents_ids=[doc.id for doc in documents],
            all_users=all_users,
            user_id=current_user.id
        )
        
        return {
            "status": "success", 
            "message": f"Запущена переиндексация {len(documents)} документов. Обработка продолжается в фоне.",
            "documents_count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при переиндексации: {str(e)}"
        )

def reindex_documents_background(documents_ids, all_users, user_id):
    """
    Фоновая функция для переиндексации документов.
    """
    logger.info(f"Начало фоновой переиндексации документов. Всего документов: {len(documents_ids)}")
    
    # Создаем новую сессию БД для этого потока
    from app.database import get_db
    db = next(get_db())
    
    try:
        # Получаем все документы
        documents = db.query(DocumentModel).filter(DocumentModel.id.in_(documents_ids)).all()
        
        # Очищаем индекс
        rag_service = get_rag_service()
        rag_service.clear_index()
        
        # Индексируем каждый документ
        reindexed_count = 0
        total_chunks = 0
        
        for idx, doc in enumerate(documents):
            progress_pct = int((idx / len(documents)) * 100)
            logger.info(f"Переиндексация документа {idx+1}/{len(documents)} ({progress_pct}%)")
            
            if not doc.content:
                logger.warning(f"Документ {doc.id} не содержит текста, пропускаем")
                continue
            
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
                
                # Сохраняем промежуточный результат после каждого документа
                db.commit()
                
            except Exception as e:
                logger.error(f"Error reindexing document {doc.id}: {str(e)}", exc_info=True)
                doc.status = "error"
                doc.error_message = str(e)
                
                # Сохраняем информацию об ошибке
                db.commit()
                
        # После завершения переиндексации всех документов
        # Еще раз явно проверяем и устанавливаем статус всех документов
        for doc_id in documents_ids:
            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if doc and doc.status == "reindexing":
                # Если документ остался в статусе reindexing, значит произошла ошибка
                doc.status = "error"
                doc.error_message = "Переиндексация не была завершена корректно"
                db.commit()
        
        logger.info(f"Завершена переиндексация документов. Обработано: {reindexed_count}/{len(documents)}, создано чанков: {total_chunks}")
            
    except Exception as e:
        logger.error(f"Общая ошибка при переиндексации документов: {str(e)}", exc_info=True)
        
        # В случае общей ошибки переводим все документы в состояние ошибки
        try:
            for doc_id in documents_ids:
                doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
                if doc and doc.status == "reindexing":
                    doc.status = "error"
                    doc.error_message = f"Ошибка при переиндексации: {str(e)}"
            db.commit()
        except Exception as commit_error:
            logger.error(f"Не удалось обновить статусы документов: {str(commit_error)}", exc_info=True)
    finally:
        db.close()

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
    Администраторы могут просматривать информацию о любом документе.
    """
    # Формируем запрос в зависимости от роли пользователя
    query = db.query(DocumentModel).filter(DocumentModel.id == document_id)
    
    # Для обычных пользователей фильтруем только их документы
    if current_user.role != "admin":
        query = query.filter(DocumentModel.user_id == current_user.id)
    
    document = query.first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Документ не найден или доступ запрещен"
        )
    
    # Получаем данные о пользователе, загрузившем документ (для админов)
    uploader = None
    uploader_name = None
    if current_user.role == "admin":
        uploader = db.query(UserModel).filter(UserModel.id == document.user_id).first()
        uploader_name = uploader.username if uploader else "Unknown"
    
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
    
    result = {
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
    
    # Добавляем информацию о владельце для администраторов
    if current_user.role == "admin":
        result["user_id"] = document.user_id
        result["uploader"] = uploader_name
    
    return result 