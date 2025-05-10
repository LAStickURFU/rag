"""
Унифицированный модуль для разбиения документов на фрагменты.
Поддерживает разные стратегии чанкинга: по символам, токенам и семантике.
"""

import re
import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path

# Импорты для LangChain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document as LCDocument

# Опциональные импорты
try:
    from transformers import AutoTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

try:
    import spacy
    from spacy.language import Language
    HAVE_SPACY = True
except ImportError:
    HAVE_SPACY = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Поддерживаемые типы контента
CONTENT_TYPES = ["text", "markdown", "html", "pdf", "code", "json", "csv", "xml", "yaml", "txt"]

# Кэш для загруженных моделей spaCy
spacy_models = {}
# Кэш для токенизаторов
tokenizers = {}


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


class UnifiedChunker:
    """
    Унифицированный класс для разбиения документов на фрагменты с несколькими стратегиями.
    """
    
    def __init__(self, 
                 mode: str = "character",  # character, token, semantic
                 chunk_size: int = 400, 
                 chunk_overlap: int = 100,
                 max_tokens: int = 512,
                 overlap_tokens: int = 20,
                 tokenizer_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 language: str = "russian",
                 spacy_model: str = "ru_core_news_md"):
        """
        Инициализация унифицированного чанкера.
        
        Args:
            mode: Режим разбиения ("character", "token", "semantic")
            chunk_size: Максимальный размер фрагмента в символах (для character mode)
            chunk_overlap: Перекрытие между соседними фрагментами в символах (для character mode)
            max_tokens: Максимальное количество токенов в чанке (для token mode)
            overlap_tokens: Перекрытие между чанками в токенах (для token mode)
            tokenizer_name: Название модели для токенизатора (для token mode)
            language: Язык документов для языкоспецифичных методов разбиения
            spacy_model: Название модели spaCy для обработки текста (для semantic mode)
        """
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer_name = tokenizer_name
        self.language = language
        self.spacy_model = spacy_model
        
        logger.info(f"Initializing UnifiedChunker in {mode} mode")
        
        # Создаем базовый рекурсивный сплиттер для текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        # Для Markdown создаем специальный сплиттер с поддержкой заголовков
        self.md_headers = [
            ("#", "Heading1"),
            ("##", "Heading2"),
            ("###", "Heading3"),
            ("####", "Heading4"),
        ]
        
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.md_headers
        )
        
        # HTML сплиттер с учетом HTML-тегов
        self.html_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "</p>\n", "</div>\n", "</section>\n", "</article>\n",
                "</p>", "</div>", "</section>", "</article>",
                "<br>\n", "<br/>", "<br />",
                "\n\n", "\n", ". ", " ", ""
            ],
            keep_separator=True
        )
        
        # Инициализируем токенизатор, если используется token mode
        self.tokenizer = None
        if mode == "token" and HAVE_TRANSFORMERS:
            self._init_tokenizer()
        
        # Инициализируем модель spaCy, если используется semantic mode
        self.nlp = None
        if mode == "semantic" and HAVE_SPACY:
            self._init_spacy()
    
    def _init_tokenizer(self):
        """Инициализирует токенизатор для token mode."""
        global tokenizers
        
        if not HAVE_TRANSFORMERS:
            logger.warning("Transformers not available. Falling back to character mode.")
            self.mode = "character"
            return
        
        # Проверяем кэш
        if self.tokenizer_name in tokenizers:
            self.tokenizer = tokenizers[self.tokenizer_name]
            logger.info(f"Using cached tokenizer: {self.tokenizer_name}")
            return
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            tokenizers[self.tokenizer_name] = self.tokenizer
            logger.info(f"Loaded tokenizer: {self.tokenizer_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_name}: {str(e)}")
            # Используем запасной токенизатор
            try:
                fallback_model = "intfloat/multilingual-e5-large"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                tokenizers[self.tokenizer_name] = self.tokenizer
                logger.info(f"Loaded fallback tokenizer: {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback tokenizer: {str(fallback_error)}")
                logger.warning("Falling back to character mode")
                self.mode = "character"
    
    def _init_spacy(self):
        """Инициализирует модель spaCy для semantic mode."""
        global spacy_models
        
        if not HAVE_SPACY:
            logger.warning("spaCy not available. Falling back to character mode.")
            self.mode = "character"
            return
        
        # Проверяем кэш
        if self.spacy_model in spacy_models:
            self.nlp = spacy_models[self.spacy_model]
            logger.info(f"Using cached spaCy model: {self.spacy_model}")
            return
        
        try:
            # Загружаем модель
            self.nlp = spacy.load(self.spacy_model)
            spacy_models[self.spacy_model] = self.nlp
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except Exception as e:
            logger.error(f"Error loading spaCy model '{self.spacy_model}': {str(e)}")
            logger.warning("Trying to download the model...")
            
            try:
                # Пробуем скачать модель
                spacy.cli.download(self.spacy_model)
                self.nlp = spacy.load(self.spacy_model)
                spacy_models[self.spacy_model] = self.nlp
                logger.info(f"Successfully downloaded and loaded spaCy model: {self.spacy_model}")
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {str(download_error)}")
                logger.warning("Falling back to character mode")
                self.mode = "character"
                # Создаем blank модель как запасной вариант
                self.nlp = spacy.blank("ru")
                spacy_models[self.spacy_model] = self.nlp
    
    def _detect_content_type(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Определяет тип содержимого для выбора подходящего разделителя.
        
        Args:
            content: Текстовое содержимое документа
            metadata: Метаданные документа, могут содержать тип контента
            
        Returns:
            Тип содержимого
        """
        # Если тип контента указан в метаданных, используем его
        if metadata and "content_type" in metadata:
            content_type = metadata["content_type"].lower()
            if content_type in CONTENT_TYPES:
                return content_type
                
        # Если у нас есть имя файла, проверяем расширение
        if metadata and "source" in metadata:
            source = metadata["source"]
            if isinstance(source, str):
                ext = Path(source).suffix.lower()
                if ext == ".md":
                    return "markdown"
                elif ext in [".html", ".htm"]:
                    return "html"
                elif ext in [".py", ".js", ".java", ".c", ".cpp", ".cs", ".php", ".rb", ".go", ".ts"]:
                    return "code"
                elif ext == ".json":
                    return "json"
                elif ext == ".csv":
                    return "csv"
                elif ext == ".xml":
                    return "xml"
                elif ext in [".yaml", ".yml"]:
                    return "yaml"
                elif ext == ".txt":
                    return "text"
        
        # Проверяем структуру содержимого
        content_sample = content.strip()[:1000]  # Проверяем первые 1000 символов
        
        # Проверка на JSON
        if content_sample.startswith('{') and ('}' in content_sample) or \
           content_sample.startswith('[') and (']' in content_sample):
            try:
                json.loads(content_sample if len(content_sample) < 1000 else content_sample + ']' 
                          if content_sample.startswith('[') else content_sample + '}')
                return "json"
            except json.JSONDecodeError:
                pass  # Не JSON, продолжаем проверки
        
        # Проверяем наличие markdown-заголовков
        if re.search(r'^#+\s+', content, re.MULTILINE):
            return "markdown"
        
        # Проверяем наличие HTML-тегов
        if re.search(r'<\/?[a-z][\s\S]*>', content):
            return "html"
        
        # По умолчанию считаем текстом
        return "text"
    
    def _split_character_based(self, content: str, content_type: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает контент на основе символов с учетом типа контента.
        
        Args:
            content: Текстовое содержимое
            content_type: Тип содержимого
            metadata: Метаданные
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        chunks = []
        
        try:
            if content_type == "markdown":
                # Сначала разбиваем по заголовкам, затем по размеру чанков
                try:
                    md_docs = self.md_splitter.split_text(content)
                    lc_chunks = self.text_splitter.split_documents(md_docs)
                except Exception as e:
                    logger.warning(f"Failed to split markdown with headers: {str(e)}")
                    # Если не удалось разбить по заголовкам, используем обычный текстовый сплиттер
                    lc_chunks = self.text_splitter.create_documents([content], [metadata])
            elif content_type == "html":
                lc_chunks = self.html_splitter.create_documents([content], [metadata])
            else:  # text и другие форматы
                lc_chunks = self.text_splitter.create_documents([content], [metadata])
            
            # Преобразуем LangChain документы в наш формат
            for i, chunk in enumerate(lc_chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["content_type"] = content_type
                
                chunks.append({
                    "text": chunk.page_content,
                    "metadata": chunk_metadata
                })
            
        except Exception as e:
            logger.error(f"Error in character-based splitting: {str(e)}")
            # Если произошла ошибка, возвращаем весь текст как один чанк
            chunks.append({
                "text": content,
                "metadata": {**metadata, "chunk_id": 0, "content_type": content_type}
            })
        
        return chunks
    
    def _split_token_based(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки на основе токенов.
        
        Args:
            content: Текст для разбиения
            metadata: Метаданные
            
        Returns:
            Список словарей {text: str, token_count: int, metadata: dict}
        """
        if not self.tokenizer:
            logger.warning("Tokenizer not initialized. Falling back to character-based splitting.")
            return self._split_character_based(content, "text", metadata)
        
        # Токенизируем текст
        tokens = self.tokenizer.encode(content)
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            # Определяем конец текущего чанка
            end = min(start + self.max_tokens, len(tokens))
            
            # Извлекаем токены для текущего чанка
            chunk_tokens = tokens[start:end]
            
            # Декодируем токены обратно в текст
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Создаем метаданные для чанка
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "content_type": "text",
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end
            })
            
            # Сохраняем чанк
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "metadata": chunk_metadata
            })
            
            # Переходим к следующему чанку с учетом перекрытия
            start += self.max_tokens - self.overlap_tokens
            chunk_id += 1
        
        return chunks
    
    def _split_semantic_based(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает текст на семантические блоки с учетом структуры и сущностей.
        
        Args:
            content: Текст для разбиения
            metadata: Метаданные
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        if not self.nlp:
            logger.warning("spaCy model not initialized. Falling back to character-based splitting.")
            return self._split_character_based(content, "text", metadata)
        
        try:
            # Разбиваем текст на параграфы
            paragraphs = re.split(r'\n\s*\n', content)
            
            chunks = []
            current_chunk = []
            current_length = 0
            chunk_id = 0
            
            # Обрабатываем каждый параграф
            for paragraph in paragraphs:
                # Если параграф слишком короткий, добавляем его целиком
                if len(paragraph) < self.chunk_size * 0.8:
                    if current_length + len(paragraph) <= self.chunk_size:
                        current_chunk.append(paragraph)
                        current_length += len(paragraph) + 2  # +2 для \n\n
                    else:
                        # Добавляем текущий чанк и начинаем новый
                        if current_chunk:
                            chunk_text = "\n\n".join(current_chunk).strip()
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = chunk_id
                            chunk_metadata["content_type"] = "text"
                            
                            chunks.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                            chunk_id += 1
                        
                        current_chunk = [paragraph]
                        current_length = len(paragraph)
                else:
                    # Для длинных параграфов используем spaCy для разбиения
                    # с учетом предложений и сущностей
                    doc = self.nlp(paragraph)
                    sentences = list(doc.sents)
                    
                    sent_chunk = []
                    sent_length = 0
                    
                    for sent in sentences:
                        if sent_length + len(sent.text) <= self.chunk_size:
                            sent_chunk.append(sent.text)
                            sent_length += len(sent.text) + 1  # +1 для пробела
                        else:
                            # Добавляем накопленные предложения как чанк
                            if sent_chunk:
                                # Сначала добавляем накопленный чанк из предыдущих параграфов
                                if current_chunk:
                                    chunk_text = "\n\n".join(current_chunk).strip()
                                    chunk_metadata = metadata.copy()
                                    chunk_metadata["chunk_id"] = chunk_id
                                    chunk_metadata["content_type"] = "text"
                                    
                                    chunks.append({
                                        "text": chunk_text,
                                        "metadata": chunk_metadata
                                    })
                                    chunk_id += 1
                                    current_chunk = []
                                    current_length = 0
                                
                                # Затем добавляем чанк из предложений
                                sent_text = " ".join(sent_chunk)
                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_id"] = chunk_id
                                chunk_metadata["content_type"] = "text"
                                
                                chunks.append({
                                    "text": sent_text,
                                    "metadata": chunk_metadata
                                })
                                chunk_id += 1
                                
                                sent_chunk = [sent.text]
                                sent_length = len(sent.text)
                            else:
                                # Если предложение слишком длинное, добавляем его целиком
                                if current_chunk:
                                    chunk_text = "\n\n".join(current_chunk).strip()
                                    chunk_metadata = metadata.copy()
                                    chunk_metadata["chunk_id"] = chunk_id
                                    chunk_metadata["content_type"] = "text"
                                    
                                    chunks.append({
                                        "text": chunk_text,
                                        "metadata": chunk_metadata
                                    })
                                    chunk_id += 1
                                    current_chunk = []
                                    current_length = 0
                                
                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_id"] = chunk_id
                                chunk_metadata["content_type"] = "text"
                                
                                chunks.append({
                                    "text": sent.text,
                                    "metadata": chunk_metadata
                                })
                                chunk_id += 1
                                sent_chunk = []
                                sent_length = 0
                    
                    # Обрабатываем оставшиеся предложения
                    if sent_chunk:
                        sent_text = " ".join(sent_chunk)
                        
                        # Проверяем, можно ли добавить к текущему чанку
                        if current_length + len(sent_text) <= self.chunk_size:
                            current_chunk.append(sent_text)
                            current_length += len(sent_text) + 2  # +2 для \n\n
                        else:
                            # Создаем новый чанк из текущих параграфов
                            if current_chunk:
                                chunk_text = "\n\n".join(current_chunk).strip()
                                chunk_metadata = metadata.copy()
                                chunk_metadata["chunk_id"] = chunk_id
                                chunk_metadata["content_type"] = "text"
                                
                                chunks.append({
                                    "text": chunk_text,
                                    "metadata": chunk_metadata
                                })
                                chunk_id += 1
                            
                            # Новый чанк из текущих предложений
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = chunk_id
                            chunk_metadata["content_type"] = "text"
                            
                            chunks.append({
                                "text": sent_text,
                                "metadata": chunk_metadata
                            })
                            chunk_id += 1
                            current_chunk = []
                            current_length = 0
            
            # Добавляем последний чанк, если есть
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk).strip()
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["content_type"] = "text"
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error in semantic-based splitting: {str(e)}")
            # В случае ошибки используем символьное разбиение
            return self._split_character_based(content, "text", metadata)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Разбивает текст на фрагменты с учетом выбранного режима.
        
        Args:
            text: Текст для разбиения
            metadata: Метаданные
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        if not text:
            return []
        
        metadata = metadata.copy() if metadata else {}
        
        # Определяем тип контента
        content_type = self._detect_content_type(text, metadata)
        metadata["content_type"] = content_type
        
        # Выбираем стратегию разбиения
        if self.mode == "token":
            return self._split_token_based(text, metadata)
        elif self.mode == "semantic":
            return self._split_semantic_based(text, metadata)
        else:  # character mode
            return self._split_character_based(text, content_type, metadata)
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает фрагменты из документа.
        
        Args:
            document: Документ для разбиения
            doc_id: Идентификатор документа
            
        Returns:
            Список фрагментов
        """
        # Подготавливаем метаданные документа
        doc_metadata = document.metadata.copy() if document.metadata else {}
        doc_metadata["doc_id"] = doc_id
        
        # Разбиваем документ на чанки
        chunks_data = self.split_text(document.content, doc_metadata)
        
        # Преобразуем в формат Chunk
        result_chunks = []
        for i, chunk_data in enumerate(chunks_data):
            # Если chunk_id уже есть в метаданных, используем его
            chunk_id = chunk_data["metadata"].get("chunk_id", i)
            
            chunk = Chunk(
                text=chunk_data["text"],
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata=chunk_data.get("metadata", {})
            )
            result_chunks.append(chunk)
        
        return result_chunks


def get_chunker(mode: str = "character", **kwargs) -> Union[UnifiedChunker, 'HierarchicalChunker']:
    """
    Создает и возвращает экземпляр чанкера с указанным режимом.
    
    Args:
        mode: Режим разбиения ("character", "token", "semantic", "hierarchical")
        **kwargs: Дополнительные параметры для чанкера
        
    Returns:
        Экземпляр UnifiedChunker или HierarchicalChunker
    """
    if mode == "hierarchical":
        # Импортируем здесь, чтобы избежать циклических импортов
        from app.chunking.hierarchical_chunker import HierarchicalChunker
        return HierarchicalChunker(**kwargs)
    else:
        return UnifiedChunker(mode=mode, **kwargs) 