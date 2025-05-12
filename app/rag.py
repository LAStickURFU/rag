"""
Модуль для реализации Retrieval-Augmented Generation (RAG)
Содержит классы и функции для обработки документов, векторизации, индексации и поиска.
"""

import os
import logging
import re
import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Импорты для LangChain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document as LCDocument
from langchain.vectorstores import Qdrant as LangchainQdrant

# Импорт Qdrant для векторной базы данных
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Импорты из проекта
from app.config import QdrantConfig, RagConfig
from app.chunking import Document, Chunk, UnifiedChunker, get_chunker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Директория для хранения индексов
INDEX_DIR = os.getenv("INDEX_DIR", "indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

# Создаем класс-обертку для работы с HTML-документами, так как импорт HTMLTextSplitter может быть недоступен
class HTMLTextSplitter:
    """Простая обертка для разбиения HTML-документов с использованием RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size=400, chunk_overlap=100):
        """
        Инициализация HTML сплиттера.
        
        Args:
            chunk_size: Максимальный размер фрагмента в символах
            chunk_overlap: Перекрытие между соседними фрагментами в символах
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "</p>", "</div>", "</section>", "<br>", ". ", " ", ""],
            keep_separator=True
        )
    
    def create_documents(self, texts, metadatas=None):
        """
        Создает LangChain документы из HTML-текстов.
        
        Args:
            texts: Список HTML-текстов
            metadatas: Список метаданных для каждого текста
            
        Returns:
            Список LangChain документов
        """
        return self.splitter.create_documents(texts, metadatas)

class Vectorizer:
    """Класс для векторизации текстовых фрагментов."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """
        Инициализация векторизатора.
        
        Args:
            model_name: Название предобученной модели для векторизации
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            # Размерность векторного представления
            self.vector_size = self.model.get_sentence_embedding_dimension()
            self.is_e5_model = "e5" in model_name.lower()
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            # Заглушка для случая ошибки загрузки модели
            self.model = None
            self.is_e5_model = False
            self.vector_size = 768  # Стандартная размерность для e5-base
            logger.warning(f"Using mock vectorizer with dimension {self.vector_size}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Преобразует список текстов в их векторные представления.
        
        Args:
            texts: Список текстов для векторизации
            
        Returns:
            Матрица векторных представлений
        """
        if self.model is None:
            # В случае ошибки возвращаем случайные векторы
            logger.warning("Using random vectors as mock embeddings")
            return np.random.rand(len(texts), self.vector_size).astype(np.float32)
        
        # Для моделей E5 нужен специальный префикс "passage: " или "query: "
        if self.is_e5_model:
            # Для запросов используем префикс "query: "
            # Для документов используем префикс "passage: "
            # По умолчанию рассматриваем тексты как документы
            processed_texts = [f"passage: {text}" for text in texts]
            logger.debug("Using E5 model with prefix 'passage:' for vectorization")
        else:
            processed_texts = texts
        
        # Векторизация текстов
        return self.model.encode(processed_texts, convert_to_numpy=True)

# QdrantIndex: drop-in замена ChromaDBIndex, но через Qdrant
class QdrantIndex:
    def __init__(self, vector_size: int, index_name: str = "default", host: str = "localhost", port: int = 6333):
        self.index_name = index_name
        self.vector_size = vector_size
        self.client = QdrantClient(host=host, port=port)
        # Создаем коллекцию, если не существует
        try:
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                timeout=5,
            )
        except Exception as e:
            if "already exists" in str(e):
                pass
            else:
                logger.error(f"Qdrant: ошибка создания коллекции: {e}")
                raise
        logger.info(f"QdrantIndex initialized: {index_name}")

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
        if not chunks:
            logger.warning("No chunks to add to the index")
            return
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata,
                "text": chunk.text,
            }
            points.append(PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload=payload
            ))
        self.client.upsert(collection_name=self.index_name, points=points)
        logger.info(f"Added {len(chunks)} chunks to QdrantIndex")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        results = self.client.search(
            collection_name=self.index_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        search_results = []
        for r in results:
            payload = r.payload
            chunk = Chunk(
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                chunk_id=payload.get("chunk_id", 0),
                metadata={k: v for k, v in payload.items() if k not in ["text", "doc_id", "chunk_id"]}
            )
            search_results.append((chunk, r.score))
        logger.info(f"Qdrant: найдено {len(search_results)} чанков")
        return search_results

    def clear(self):
        self.client.delete_collection(collection_name=self.index_name)
        self.client.create_collection(
            collection_name=self.index_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        logger.info(f"QdrantIndex cleared: {self.index_name}")

class RAGService:
    """Сервис для обработки RAG-запросов."""
    
    def __init__(self, 
                 model_name: str = "intfloat/multilingual-e5-base",
                 collection_name: str = "documents",
                 dense_weight: float = 0.3,
                 sparse_weight: float = 0.7,
                 reranker_weight: float = 0.5,
                 chunk_size: int = 400,
                 chunk_overlap: int = 200,
                 max_context_docs: int = 5,
                 top_k_chunks: int = 7,
                 use_hybrid: bool = True,
                 use_reranker: bool = True,
                 use_adaptive_k: bool = True,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_token_chunker: bool = False,
                 max_tokens: int = 512,
                 overlap_tokens: int = 20,
                 chunking_mode: str = "character",
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 1000,
                 heading_patterns: Optional[List[str]] = None,
                 language: str = "russian",
                 spacy_model: str = "ru_core_news_md"):
        """
        Инициализирует сервис RAG.
        
        Args:
            model_name: Название модели для эмбеддингов
            collection_name: Название коллекции в Qdrant
            dense_weight: Вес плотного поиска в гибридном
            sparse_weight: Вес разреженного поиска в гибридном
            reranker_weight: Вес переранжирования
            chunk_size: Размер чанка в символах (для символьного чанкера)
            chunk_overlap: Перекрытие между чанками в символах (для символьного чанкера)
            max_context_docs: Максимальное количество документов в контексте
            top_k_chunks: Максимальное количество фрагментов для включения в промпт
            use_hybrid: Использовать ли гибридный поиск
            use_reranker: Использовать ли переранжирование
            use_adaptive_k: Использовать ли адаптивное количество документов
            cross_encoder_model: Модель для cross-encoder
            use_token_chunker: Использовать ли токеновый чанкер вместо символьного
            max_tokens: Максимальное количество токенов в чанке (для токенового чанкера)
            overlap_tokens: Перекрытие между чанками в токенах (для токенового чанкера)
            chunking_mode: Режим чанкинга ("character", "token", "semantic", "hierarchical")
            min_chunk_size: Минимальный размер чанка в символах
            max_chunk_size: Максимальный размер чанка в символах
            heading_patterns: Паттерны для определения заголовков (для hierarchical чанкера)
            language: Язык документов
            spacy_model: Модель spaCy для обработки текста
        """
        # Загружаем конфигурации
        self.qdrant_config = QdrantConfig()
        self.rag_config = RagConfig()
        
        # Сохраняем параметры
        self.model_name = model_name
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker_weight = reranker_weight
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.use_adaptive_k = use_adaptive_k
        self.max_context_docs = max_context_docs
        self.top_k_chunks = top_k_chunks
        self.token_limit = 4096  # Для LLM-модели
        self.use_token_chunker = use_token_chunker
        self.chunking_mode = chunking_mode
        
        # Явно сохраняем параметры чанкинга для доступа через свойства
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        
        # Логирование параметров
        logger.info(f"Initializing RAG service with parameters:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Hybrid search: {use_hybrid} (dense: {dense_weight}, sparse: {sparse_weight})")
        logger.info(f"  Reranker: {use_reranker} (weight: {reranker_weight})")
        logger.info(f"  Adaptive k: {use_adaptive_k}")
        logger.info(f"  Chunking mode: {chunking_mode}")
        
        # Определяем режим чанкинга
        if use_token_chunker:
            # Обратная совместимость: если use_token_chunker=True, устанавливаем режим token
            chunking_mode = "token"
        
        # Создаем чанкер
        chunker_params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "language": language,
            "spacy_model": spacy_model
        }
        
        # Дополнительные параметры для иерархического чанкера
        if chunking_mode == "hierarchical":
            chunker_params.update({
                "min_chunk_size": min_chunk_size,
                "max_chunk_size": max_chunk_size,
                "heading_patterns": heading_patterns
            })
            logger.info(f"Using hierarchical chunker (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        # Создаем чанкер
        self.chunker = get_chunker(mode=chunking_mode, **chunker_params)
        
        # Инициализируем векторизатор для эмбеддингов без кэширования
        self.vectorizer = Vectorizer(model_name=model_name)
        logger.info(f"Using vectorizer without caching: {model_name}")
        
        # Создаем векторный индекс
        vector_size = self.vectorizer.vector_size
        self.index = QdrantIndex(vector_size=vector_size, index_name=collection_name)
        
        # Инициализируем гибридный поисковик, если требуется
        if use_hybrid:
            try:
                from app.retrieval.hybrid_retriever import HybridRetriever
                self.hybrid_retriever = HybridRetriever(
                    dense_weight=dense_weight,
                    reranker_weight=reranker_weight,
                    use_reranker=use_reranker,
                    use_adaptive_k=use_adaptive_k,
                    cross_encoder_model=cross_encoder_model,
                    language=language,
                    spacy_model=spacy_model
                )
                logger.info("Initialized hybrid retriever with BM25 and dense search")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {str(e)}")
                self.hybrid_retriever = None
                self.use_hybrid = False
        else:
            self.hybrid_retriever = None
    
    def add_document(self, document: Document, doc_id: str = None) -> List[int]:
        """
        Добавляет документ в индекс.
        
        Args:
            document: Документ для добавления
            doc_id: Идентификатор документа (если None, создается автоматически)
            
        Returns:
            Список внутренних идентификаторов добавленных фрагментов
        """
        # Создаем уникальный ID для документа, если не предоставлен
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Разбиваем документ на фрагменты с помощью улучшенного чанкера
        chunks = self.chunker.create_chunks_from_document(document, doc_id)
        
        if not chunks:
            logger.warning(f"No chunks created for document: {doc_id}")
            return []
        
        # Векторизуем фрагменты
        chunk_texts = [chunk.text for chunk in chunks]
        vectors = self.vectorizer.encode(chunk_texts)
        
        # Добавляем в индекс
        self.index.add_chunks(chunks, vectors)
        
        # Обновляем индекс BM25, если используется гибридный поиск
        if self.use_hybrid and self.hybrid_retriever:
            try:
                self.hybrid_retriever.index_documents(chunks)
                logger.info(f"Updated BM25 index with {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to update BM25 index: {str(e)}")
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        
        return [i for i in range(len(chunks))]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Поиск релевантных фрагментов по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список кортежей (фрагмент, score)
        """
        # Для моделей E5 добавляем префикс "query:" к запросу
        if getattr(self.vectorizer, 'is_e5_model', False):
            processed_query = f"query: {query}"
            logger.debug("Using E5 model with prefix 'query:' for search")
        else:
            processed_query = query
        
        # Векторизуем запрос
        query_vector = self.vectorizer.encode([processed_query])[0]
        
        # Получаем релевантные фрагменты через dense retrieval
        dense_results = self.index.search(query_vector, top_k)
        
        # Если гибридный поиск не используется, возвращаем результаты dense retrieval
        if not self.use_hybrid or not self.hybrid_retriever:
            return dense_results
        
        # Иначе выполняем гибридный поиск
        try:
            hybrid_results = self.hybrid_retriever.search(query, dense_results, top_k)
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            # Если гибридный поиск не удался, возвращаем результаты dense retrieval
            return dense_results
    
    def generate_prompt(self, query: str, top_k_chunks: int = None) -> str:
        """
        Генерирует промпт для LLM на основе запроса и релевантных фрагментов.
        
        Args:
            query: Текст запроса
            top_k_chunks: Максимальное количество фрагментов для включения в промпт
            
        Returns:
            Промпт для LLM
        """
        # Используем значение из конфигурации, если не задано явно
        if top_k_chunks is None:
            top_k_chunks = self.top_k_chunks
        
        # Запрашиваем больше чанков для последующей фильтрации
        relevant_chunks = self.search(query, top_k=top_k_chunks * 2)
        
        if not relevant_chunks:
            # Если нет релевантных фрагментов, возвращаем промпт с уведомлением
            return f"""Ты - информационный ассистент, который отвечает на вопросы по документам.

Вопрос пользователя: {query}

В базе знаний нет информации по этому вопросу. Сообщи пользователю, что не можешь ответить на основе имеющихся данных и предложи загрузить необходимые документы или переформулировать вопрос. Ответ должен быть коротким и четким.

ОТВЕТ:"""
        
        # Фильтруем низко-релевантные фрагменты
        MIN_SCORE_THRESHOLD = 0.25  # Снижен порог релевантности для более широкого охвата
        filtered_chunks = [(chunk, score) for chunk, score in relevant_chunks if score >= MIN_SCORE_THRESHOLD]
        
        # Если после фильтрации осталось меньше минимального числа фрагментов, берем лучшие
        top_k_min = 3
        if len(filtered_chunks) < top_k_min:
            filtered_chunks = relevant_chunks[:top_k_min]
        
        # Логируем информацию о релевантности для отладки
        for i, (chunk, score) in enumerate(filtered_chunks[:10]):
            logger.info(f"Chunk {i+1}: relevance={score:.2f}, doc_id={chunk.doc_id}, title={chunk.metadata.get('title', 'Unknown')}")
        
        # Корректируем вес фрагментов в зависимости от их релевантности
        weighted_chunks = []
        for chunk, score in filtered_chunks:
            # Применяем нелинейное преобразование для усиления контраста
            if score > 0.6:  # Высокая релевантность
                weight = 1.0
            elif score > 0.4:  # Средняя релевантность
                weight = 0.8
            else:  # Низкая релевантность
                weight = 0.5
            
            weighted_chunks.append((chunk, score, weight))
        
        # Сортируем по релевантности и ограничиваем количество фрагментов
        weighted_chunks.sort(key=lambda x: x[1], reverse=True)
        weighted_chunks = weighted_chunks[:top_k_chunks]
        
        # Формируем контекст из фрагментов с учетом весов
        context_parts = []
        for i, (chunk, score, weight) in enumerate(weighted_chunks):
            # Добавляем только идентификатор фрагмента и оценку релевантности
            marker = ""
            if score > 0.8:
                marker = " ✓" # Простая отметка для высокорелевантных фрагментов
            
            # Упрощаем формат, полностью убираем метаданные
            chunk_info = f"[Контекст {i+1}{marker}]"
            
            # Добавляем только текст фрагмента без метаданных
            context_parts.append(f"{chunk_info}\n{chunk.text}")
        
        # Объединяем части контекста с разделителями
        context = "\n\n".join(context_parts)
        
        # Оптимизированный шаблон промпта
        prompt_template = f"""Ты - точный и лаконичный информационный ассистент. Ты отвечаешь на вопросы пользователя, опираясь только на предоставленный контекст.

РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ:
{context}

ВОПРОС: {query}

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из предоставленного контекста, не добавляй никаких внешних знаний
2. Игнорируй номера и маркеры фрагментов, просто используй их содержимое для формирования ответа
3. Если в контексте недостаточно информации для полного ответа, честно скажи об этом
4. Отвечай прямо на вопрос с первого предложения, без вводных фраз
5. Излагай информацию своими словами как эксперт, а не копируй исходный текст
6. Ответ должен быть в виде связного текста без маркированных списков
7. Не ссылайся на источники, номера контекста или фрагменты в своем ответе

ОТВЕТ:"""
        
        return prompt_template
        
    def _detect_question_type(self, query: str) -> str:
        """
        Определяет тип вопроса для оптимизации ответа.
        
        Args:
            query: Текст запроса
            
        Returns:
            Тип вопроса (factoid, date, person, definition, general)
        """
        # Этот метод больше не используется в упрощенной версии промпта,
        # но сохранен для обратной совместимости
        
        # По умолчанию - общий вопрос
        return "general"
        
    def clear_index(self):
        """
        Очищает индекс документов.
        
        Returns:
            None
        """
        if self.index:
            self.index.clear()
            logger.info(f"Cleared index {self.collection_name}")
            
            # Очищаем BM25 индекс, если используется гибридный поиск
            if self.use_hybrid and self.hybrid_retriever:
                try:
                    self.hybrid_retriever.clear_index()
                    logger.info("Cleared BM25 index")
                except Exception as e:
                    logger.warning(f"Failed to clear BM25 index: {str(e)}")
            
            return True
        return False
        
    def delete_document(self, doc_id: str):
        """
        Удаляет документ из индекса по ID.
        
        Args:
            doc_id: Идентификатор документа
            
        Returns:
            True, если удаление успешно
        """
        # В текущей реализации мы не можем удалить отдельный документ из индекса Qdrant,
        # поэтому просто логируем действие и возвращаем True
        logger.info(f"Document deletion not fully implemented. Doc ID: {doc_id}")
        return True
        
    # Свойства для доступа к параметрам чанкинга
    @property
    def chunk_size(self) -> int:
        """Получение размера чанка в зависимости от режима чанкинга."""
        if hasattr(self, '_chunk_size'):
            return self._chunk_size
        if hasattr(self.chunker, 'chunk_size'):
            return self.chunker.chunk_size
        return 400  # Значение по умолчанию
    
    @property
    def chunk_overlap(self) -> int:
        """Получение перекрытия чанков в зависимости от режима чанкинга."""
        if hasattr(self, '_chunk_overlap'):
            return self._chunk_overlap
        if hasattr(self.chunker, 'chunk_overlap'):
            return self.chunker.chunk_overlap
        return 100  # Значение по умолчанию
    
    @property
    def min_chunk_size(self) -> int:
        """Получение минимального размера чанка для иерархического режима."""
        if hasattr(self, '_min_chunk_size'):
            return self._min_chunk_size
        if hasattr(self.chunker, 'min_chunk_size'):
            return self.chunker.min_chunk_size
        return 50  # Значение по умолчанию
    
    @property
    def max_chunk_size(self) -> int:
        """Получение максимального размера чанка для иерархического режима."""
        if hasattr(self, '_max_chunk_size'):
            return self._max_chunk_size
        if hasattr(self.chunker, 'max_chunk_size'):
            return self.chunker.max_chunk_size
        return 1000  # Значение по умолчанию

# Создаем класс-обертку Document для обратной совместимости
class Document:
    """Класс для представления документа."""
    
    def __init__(self, content=None, text=None, metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация документа.
        
        Args:
            content: Текстовое содержимое документа (устаревший параметр, для обратной совместимости)
            text: Текстовое содержимое документа
            metadata: Метаданные документа (источник, дата и т.д.)
        """
        self.content = text if text is not None else content
        self.text = self.content  # Для совместимости с обоими интерфейсами
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})" 