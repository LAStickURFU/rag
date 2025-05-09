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

# Импорт Qdrant для векторной базы данных
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

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

class Document:
    """Класс для представления документа."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация документа.
        
        Args:
            content: Текстовое содержимое документа
            metadata: Метаданные документа (источник, дата и т.д.)
        """
        self.content = content
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"Document(content={self.content[:50]}..., metadata={self.metadata})"

class Chunk:
    """Класс для представления фрагмента документа."""
    
    def __init__(self, text: str, doc_id: str, chunk_id: int, metadata: Optional[Dict[str, Any]] = None):
        """
        Инициализация фрагмента.
        
        Args:
            text: Текстовое содержимое фрагмента
            doc_id: Идентификатор исходного документа
            chunk_id: Номер фрагмента в документе
            metadata: Метаданные фрагмента
        """
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"Chunk(text={self.text[:50]}..., doc_id={self.doc_id}, chunk_id={self.chunk_id})"

class SmartChunker:
    """Улучшенный класс для разбиения документов на фрагменты с использованием LangChain."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        """
        Инициализация умного чанкера.
        
        Args:
            chunk_size: Максимальный размер фрагмента в символах
            chunk_overlap: Перекрытие между соседними фрагментами в символах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Создаем различные разделители текста для разных типов контента
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        # Заголовки для markdown, используются для более умного разбиения документов
        self.md_headers = [
            ("#", "Heading1"),
            ("##", "Heading2"),
            ("###", "Heading3"),
            ("####", "Heading4"),
        ]
        
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.md_headers
        )
        
        self.html_splitter = HTMLTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _detect_content_type(self, content: str) -> str:
        """
        Определяет тип содержимого для выбора подходящего разделителя.
        
        Args:
            content: Текстовое содержимое документа
            
        Returns:
            Тип содержимого ("markdown", "html", "text")
        """
        # Проверяем наличие markdown-заголовков
        if re.search(r'^#+\s+', content, re.MULTILINE):
            return "markdown"
        
        # Проверяем наличие HTML-тегов
        if re.search(r'<\/?[a-z][\s\S]*>', content):
            return "html"
        
        # По умолчанию считаем текстом
        return "text"
    
    def _split_by_content_type(self, content: str, content_type: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        """
        Разбивает контент в зависимости от типа.
        
        Args:
            content: Текстовое содержимое
            content_type: Тип содержимого
            metadata: Метаданные документа
            
        Returns:
            Список фрагментов в формате LangChain
        """
        if content_type == "markdown":
            # Сначала разбиваем по заголовкам, затем по размеру чанков
            md_docs = self.md_splitter.split_text(content)
            return self.text_splitter.split_documents(md_docs)
        
        elif content_type == "html":
            return self.html_splitter.create_documents([content], [metadata])
        
        else:  # text
            return self.text_splitter.create_documents([content], [metadata])
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Умное разбиение текста на фрагменты.
        
        Args:
            text: Исходный текст
            metadata: Метаданные для контекста
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        metadata = metadata or {}
        content_type = self._detect_content_type(text)
        
        # Получаем LangChain документы
        lc_docs = self._split_by_content_type(text, content_type, metadata)
        
        # Конвертируем в наш формат
        results = []
        for i, doc in enumerate(lc_docs):
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["content_type"] = content_type
            
            # Добавляем информацию о заголовках для markdown
            if content_type == "markdown" and "Heading1" in chunk_metadata:
                chunk_metadata["heading"] = chunk_metadata.get("Heading1", "")
                if "Heading2" in chunk_metadata:
                    chunk_metadata["heading"] += " > " + chunk_metadata.get("Heading2", "")
                if "Heading3" in chunk_metadata:
                    chunk_metadata["heading"] += " > " + chunk_metadata.get("Heading3", "")
            
            results.append({
                "text": doc.page_content,
                "metadata": chunk_metadata
            })
        
        return results
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает фрагменты из документа с умным разбиением.
        
        Args:
            document: Документ для разбиения
            doc_id: Идентификатор документа
            
        Returns:
            Список фрагментов
        """
        # Подготавливаем метаданные документа
        doc_metadata = document.metadata.copy()
        doc_metadata["doc_id"] = doc_id
        
        # Умное разбиение с учетом структуры документа
        chunks_data = self.split_text(document.content, doc_metadata)
        
        # Преобразуем в формат Chunk
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk = Chunk(
                text=chunk_data["text"],
                doc_id=doc_id,
                chunk_id=i,
                metadata=chunk_data["metadata"]
            )
            chunks.append(chunk)
        
        return chunks

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
    """Служба для реализации Retrieval-Augmented Generation."""
    
    def __init__(
        self, 
        index_name: str = "default",
        model_name: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 400,  # Увеличиваем размер чанка до 400
        chunk_overlap: int = 100,  # Перекрытие между фрагментами
        use_hybrid_search: bool = True,
        use_reranker: bool = True,
        dense_weight: float = 0.7,
        reranker_weight: float = 0.5,
        use_adaptive_k: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        language: str = "russian",
        spacy_model: str = "ru_core_news_md"
    ):
        """
        Инициализация RAG сервиса.
        
        Args:
            index_name: Имя индекса для хранения векторов
            model_name: Название модели для векторизации
            chunk_size: Размер фрагмента документа
            chunk_overlap: Перекрытие между фрагментами
            use_hybrid_search: Использовать ли гибридный поиск
            use_reranker: Использовать ли переранжирование
            dense_weight: Вес dense retrieval в гибридном поиске (0-1)
            reranker_weight: Вес reranker в итоговом ранжировании (0-1)
            use_adaptive_k: Использовать ли адаптивное количество документов
            cross_encoder_model: Название модели CrossEncoder
            language: Язык для BM25 и NLTK
            spacy_model: Название модели spaCy для обработки текста
        """
        self.index_name = index_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        self.dense_weight = dense_weight
        self.reranker_weight = reranker_weight
        self.use_adaptive_k = use_adaptive_k
        self.language = language
        self.spacy_model = spacy_model
        
        # Инициализируем улучшенный чанкер с резервными методами
        from app.chunking.robust_chunker import RobustChunker
        self.chunker = RobustChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language,
            spacy_model=spacy_model
        )
        
        # Инициализируем векторизатор для эмбеддингов
        self.vectorizer = Vectorizer(model_name=model_name)
        
        # Создаем векторный индекс
        vector_size = self.vectorizer.vector_size
        self.index = QdrantIndex(vector_size=vector_size, index_name=index_name)
        
        # Инициализируем гибридный поисковик, если требуется
        if use_hybrid_search:
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
                logger.info("Initialized hybrid retriever with BM25 and dense search (spaCy-only)")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {str(e)}")
                self.hybrid_retriever = None
                self.use_hybrid_search = False
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
        if self.use_hybrid_search and self.hybrid_retriever:
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
        if not self.use_hybrid_search or not self.hybrid_retriever:
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
            top_k_chunks: Количество фрагментов для включения в промпт
            
        Returns:
            Промпт для LLM
        """
        # Получаем релевантные фрагменты
        relevant_chunks = self.search(query, top_k=top_k_chunks)
        
        if not relevant_chunks:
            # Если нет релевантных фрагментов, возвращаем промпт с уведомлением
            return f"""Ты - ассистент с фокусом на точность. Пользователь задал вопрос, но в индексе нет релевантной информации.

Вопрос пользователя: {query}

В базе знаний нет информации по этому вопросу. Сообщи пользователю, что не можешь ответить на основе имеющихся данных. 
Предложи пользователю загрузить соответствующие документы или переформулировать вопрос.

ОТВЕТ:"""
        
        # Формируем контекст из фрагментов
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks):
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
        
        # Улучшенный шаблон промпта
        prompt_template = f"""Ты - русскоязычный ассистент, специализирующийся на предоставлении точной информации на основе контекста.

ИНСТРУКЦИИ:
1. Используй ТОЛЬКО информацию из предоставленных фрагментов документов.
2. Отвечай ИСКЛЮЧИТЕЛЬНО на русском языке, с правильной грамматикой и пунктуацией.
3. Если информации недостаточно или она отсутствует в контексте, честно признай это.
4. Не используй внешние знания или предположения, не основанные на контексте.
5. Стремись к точности и краткости, избегая повторов.
6. Не упоминай в ответе номера или идентификаторы фрагментов.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""
        
        return prompt_template 