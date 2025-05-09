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
from app.chunking.robust_chunker import RobustChunker, Document, Chunk
from app.chunking.token_chunker import TokenChunker, get_default_token_chunker

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

class VectorizerWithCache:
    """Класс для векторизации текстовых фрагментов с кэшированием."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", cache_dir: str = "cache/embeddings"):
        """
        Инициализация векторизатора с кэшированием.
        
        Args:
            model_name: Название предобученной модели для векторизации
            cache_dir: Директория для кэширования эмбеддингов
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Загружаем модель
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            self.is_e5_model = "e5" in model_name.lower()
            logger.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {str(e)}")
            self.model = None
            self.is_e5_model = False
            self.vector_size = 768  # Стандартная размерность для e5-base
            logger.warning(f"Using mock vectorizer with dimension {self.vector_size}")
            
        # Инициализируем кэш
        self.cache = {}
        self._load_cache()
    
    def _get_cache_path(self, text_hash: str) -> str:
        """
        Получает путь для кэширования эмбеддинга.
        
        Args:
            text_hash: Хэш текста
            
        Returns:
            Путь к файлу кэша
        """
        return os.path.join(self.cache_dir, f"{text_hash}.npy")
    
    def _load_cache(self):
        """Загружает существующий кэш при инициализации."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_files = os.listdir(self.cache_dir)
            for cache_file in cache_files:
                if cache_file.endswith(".npy"):
                    text_hash = cache_file[:-4]  # Убираем расширение .npy
                    self.cache[text_hash] = True  # Помечаем как доступный в кэше
            logger.info(f"Loaded {len(self.cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Error loading embedding cache: {str(e)}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Получает хэш текста для кэширования.
        
        Args:
            text: Исходный текст
            
        Returns:
            Хэш текста
        """
        import hashlib
        # Используем MD5 как быстрый хэш для текста
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Преобразует список текстов в их векторные представления с кэшированием.
        
        Args:
            texts: Список текстов для векторизации
            use_cache: Использовать ли кэширование
            
        Returns:
            Матрица векторных представлений
        """
        if self.model is None:
            # В случае ошибки возвращаем случайные векторы
            logger.warning("Using random vectors as mock embeddings")
            return np.random.rand(len(texts), self.vector_size).astype(np.float32)
        
        # Проверяем кэш для каждого текста
        result_vectors = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        texts_to_encode = []
        indices_to_encode = []
        
        # Если E5 модель, добавляем префикс
        processed_texts = []
        for text in texts:
            if self.is_e5_model:
                processed_texts.append(f"passage: {text}")
            else:
                processed_texts.append(text)
        
        for i, text in enumerate(processed_texts):
            if not use_cache:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                continue
                
            text_hash = self._get_text_hash(text)
            cache_path = self._get_cache_path(text_hash)
            
            if os.path.exists(cache_path):
                # Загружаем из кэша
                try:
                    cached_vector = np.load(cache_path)
                    result_vectors[i] = cached_vector
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {str(e)}")
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Если есть тексты для кодирования
        if texts_to_encode:
            cache_hit_rate = 1.0 - (len(texts_to_encode) / len(texts)) if texts else 0
            logger.info(f"Encoding {len(texts_to_encode)} texts (cache hit rate: {cache_hit_rate:.2f})")
            
            # Векторизуем тексты
            new_vectors = self.model.encode(texts_to_encode, convert_to_numpy=True)
            
            # Сохраняем результаты и кэшируем
            for idx, (text_idx, text) in enumerate(zip(indices_to_encode, texts_to_encode)):
                result_vectors[text_idx] = new_vectors[idx]
                
                # Кэшируем если нужно
                if use_cache:
                    text_hash = self._get_text_hash(text)
                    cache_path = self._get_cache_path(text_hash)
                    try:
                        np.save(cache_path, new_vectors[idx])
                        self.cache[text_hash] = True
                    except Exception as e:
                        logger.warning(f"Error caching embedding: {str(e)}")
        
        return result_vectors

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

class Chunker:
    """Класс для разбиения документов на фрагменты."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """
        Инициализация чанкера.
        
        Args:
            chunk_size: Максимальный размер фрагмента в символах
            chunk_overlap: Перекрытие между соседними фрагментами в символах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Разбивает текст на фрагменты.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список фрагментов текста
        """
        # Разбиваем текст на параграфы
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Если параграф слишком большой, разбиваем на предложения
            if len(paragraph) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += (sentence + " ")
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                if len(current_chunk) + len(paragraph) <= self.chunk_size:
                    current_chunk += (paragraph + "\n\n")
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает фрагменты из документа.
        
        Args:
            document: Документ для разбиения
            doc_id: Идентификатор документа
            
        Returns:
            Список фрагментов
        """
        text_chunks = self.split_text(document.content)
        chunks = []
        
        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                text=text,
                doc_id=doc_id,
                chunk_id=i,
                metadata={**document.metadata, "chunk_id": i}
            )
            chunks.append(chunk)
        
        return chunks

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
                 use_hybrid: bool = True,
                 use_reranker: bool = True,
                 use_adaptive_k: bool = True,
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 cache_embeddings: bool = True,
                 use_token_chunker: bool = True,
                 max_tokens: int = 512,
                 overlap_tokens: int = 20):
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
            use_hybrid: Использовать ли гибридный поиск
            use_reranker: Использовать ли переранжирование
            use_adaptive_k: Использовать ли адаптивное количество документов
            cross_encoder_model: Модель для cross-encoder
            cache_embeddings: Кэшировать ли эмбеддинги
            use_token_chunker: Использовать ли токеновый чанкер вместо символьного
            max_tokens: Максимальное количество токенов в чанке (для токенового чанкера)
            overlap_tokens: Перекрытие между чанками в токенах (для токенового чанкера)
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
        self.token_limit = 4096  # Для LLM-модели
        self.use_token_chunker = use_token_chunker
        
        # Логирование параметров
        logger.info(f"Initializing RAG service with parameters:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Hybrid search: {use_hybrid} (dense: {dense_weight}, sparse: {sparse_weight})")
        logger.info(f"  Reranker: {use_reranker} (weight: {reranker_weight})")
        logger.info(f"  Adaptive k: {use_adaptive_k}")
        logger.info(f"  Token-based chunker: {use_token_chunker}")
        
        # Создаем чанкер в зависимости от выбранного типа
        if use_token_chunker:
            logger.info(f"Using token-based chunker (max_tokens={max_tokens}, overlap={overlap_tokens})")
            self.chunker = TokenChunker(
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens
            )
        else:
            logger.info(f"Using character-based chunker (chunk_size={chunk_size}, overlap={chunk_overlap})")
            self.chunker = RobustChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Инициализируем векторизатор для эмбеддингов - используем кэширующую версию
        if cache_embeddings:
            self.vectorizer = VectorizerWithCache(model_name=model_name)
            logger.info(f"Using vectorizer with caching: {model_name}")
        else:
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
                    language=self.rag_config.language,
                    spacy_model=self.rag_config.spacy_model
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
    
    def generate_prompt(self, query: str, top_k_chunks: int = 5) -> str:
        """
        Генерирует промпт для LLM на основе запроса и релевантных фрагментов.
        
        Args:
            query: Текст запроса
            top_k_chunks: Максимальное количество фрагментов для включения в промпт
            
        Returns:
            Промпт для LLM
        """
        # Запрашиваем больше чанков для последующей фильтрации
        relevant_chunks = self.search(query, top_k=top_k_chunks * 2)
        
        if not relevant_chunks:
            # Если нет релевантных фрагментов, возвращаем промпт с уведомлением
            return f"""Ты - ассистент с фокусом на точность. Пользователь задал вопрос, но в индексе нет релевантной информации.

Вопрос пользователя: {query}

В базе знаний нет информации по этому вопросу. Сообщи пользователю, что не можешь ответить на основе имеющихся данных. 
Предложи пользователю загрузить соответствующие документы или переформулировать вопрос.

ОТВЕТ:"""
        
        # Фильтруем низко-релевантные фрагменты
        MIN_SCORE_THRESHOLD = 0.45  # Минимальный порог релевантности
        filtered_chunks = [(chunk, score) for chunk, score in relevant_chunks if score >= MIN_SCORE_THRESHOLD]
        
        # Если после фильтрации осталось меньше минимального числа фрагментов, берем лучшие
        top_k_min = 2
        if len(filtered_chunks) < top_k_min:
            filtered_chunks = relevant_chunks[:top_k_min]
        
        # Ограничиваем максимальное количество фрагментов
        filtered_chunks = filtered_chunks[:top_k_chunks]
        
        # Сортируем фрагменты по релевантности перед формированием контекста
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Формируем контекст из фрагментов
        context_parts = []
        for i, (chunk, score) in enumerate(filtered_chunks):
            # Добавляем идентификатор фрагмента и оценку релевантности
            chunk_info = f"[Фрагмент {i+1}, релевантность: {score:.2f}]"
            
            # Добавляем информацию о заголовке, если есть
            heading = chunk.metadata.get("heading", "")
            if heading:
                chunk_info += f" {heading}"
            
            # Добавляем источник, если есть
            source = chunk.metadata.get("source", "")
            title = chunk.metadata.get("title", "")
            if title and source:
                chunk_info += f" | {title} ({source})"
            elif title:
                chunk_info += f" | {title}"
            elif source:
                chunk_info += f" | {source}"
            
            # Добавляем текст фрагмента с идентификатором и метаданными
            context_parts.append(f"{chunk_info}\n{chunk.text}")
        
        # Объединяем части контекста с разделителями
        context = "\n\n".join(context_parts)
        
        # Определяем тип вопроса для оптимизации формата ответа
        question_type = self._detect_question_type(query)
        
        # Дополнительные инструкции на основе типа вопроса
        type_specific_instructions = ""
        if question_type == "factoid":
            type_specific_instructions = "Дай ТОЛЬКО конкретный факт, без вводных фраз и объяснений. Одно предложение максимум."
        elif question_type == "date":
            type_specific_instructions = "Укажи ТОЛЬКО дату или временной период, без лишних слов."
        elif question_type == "person":
            type_specific_instructions = "Укажи ТОЛЬКО имя человека или организации в ответе, без дополнительных комментариев."
        elif question_type == "definition":
            type_specific_instructions = "Дай краткое и точное определение, не более 1-2 предложений."
        
        # Улучшенный шаблон промпта для максимизации метрики answer_similarity
        prompt_template = f"""Ты - русскоязычный ассистент, специализирующийся на предоставлении точной информации на основе контекста.

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из предоставленных фрагментов документов.
2. Дай КРАТКИЙ и ТОЧНЫЙ ответ на вопрос, основанный строго на предоставленном контексте.
3. {type_specific_instructions or "Если вопрос требует конкретного факта (имя, дата, число), приведи ТОЛЬКО этот факт без лишних слов."}
4. НЕ используй вводные фразы типа "Согласно контексту" или "На основе предоставленной информации".
5. Если информации недостаточно, кратко укажи на это, не придумывая никаких деталей.
6. Отвечай точными формулировками из контекста, когда это возможно.
7. Избегай обобщений, придерживайся конкретных фактов из контекста.
8. Если разные фрагменты противоречат друг другу, укажи это и приведи оба варианта.
9. НЕ упоминай номера фрагментов или их метаданные в своем ответе.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

КРАТКИЙ ОТВЕТ:"""
        
        return prompt_template
        
    def _detect_question_type(self, query: str) -> str:
        """
        Определяет тип вопроса для оптимизации ответа.
        
        Args:
            query: Текст запроса
            
        Returns:
            Тип вопроса (factoid, date, person, definition, general)
        """
        # Приводим к нижнему регистру для анализа
        query_lower = query.lower()
        
        # Проверяем на factoid-вопросы (конкретные факты)
        factoid_patterns = ["что такое", "что представляет собой", "что означает", 
                           "что является", "какой", "сколько", "где находится"]
        
        # Проверяем на вопросы о датах
        date_patterns = ["когда", "в каком году", "в каком месяце", "дата", "период"]
        
        # Проверяем на вопросы о персонах
        person_patterns = ["кто", "чье имя", "какой человек", "какая организация"]
        
        # Проверяем на запросы определений
        definition_patterns = ["что такое", "определение", "термин", "определить", "что означает"]
        
        # Проверяем паттерны
        for pattern in factoid_patterns:
            if pattern in query_lower:
                return "factoid"
                
        for pattern in date_patterns:
            if pattern in query_lower:
                return "date"
                
        for pattern in person_patterns:
            if pattern in query_lower:
                return "person"
                
        for pattern in definition_patterns:
            if pattern in query_lower:
                return "definition"
                
        # По умолчанию - общий вопрос
        return "general" 