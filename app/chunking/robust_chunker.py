"""
Модуль для улучшенного разбиения документов на фрагменты с механизмами резервирования.
"""

import re
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Импорты для LangChain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document as LCDocument

# Импорт spaCy вместо NLTK для более качественной обработки русских текстов
import spacy
from spacy.language import Language

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Поддерживаемые типы контента
CONTENT_TYPES = ["text", "markdown", "html", "pdf", "code"]

# Кэш для загруженной модели spaCy
spacy_models = {}


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


class RobustChunker:
    """
    Улучшенный класс для разбиения документов на фрагменты с резервными механизмами.
    """
    
    def __init__(self, 
                chunk_size: int = 400, 
                chunk_overlap: int = 100,
                language: str = "russian",
                spacy_model: str = "ru_core_news_md"):
        """
        Инициализация улучшенного чанкера.
        
        Args:
            chunk_size: Максимальный размер фрагмента в символах
            chunk_overlap: Перекрытие между соседними фрагментами в символах
            language: Язык документов для языкоспецифичных методов разбиения
            spacy_model: Название модели spaCy для обработки текста
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.spacy_model = spacy_model
        
        # Загружаем модель spaCy
        self.nlp = self._get_spacy_model(spacy_model)
        
        # Создаем основные разделители текста для разных типов контента
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        # Markdown заголовки для структурированного разбиения
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
        
        # Сплиттер для кода с учетом синтаксиса программирования
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\nclass ", "\ndef ", "\nfunction ", "\nif ", "\nfor ", "\nwhile ",
                "\n\n", "\n", "; ", " ", ""
            ],
            keep_separator=True
        )
    
    def _get_spacy_model(self, model_name: str) -> Language:
        """
        Получает или загружает модель spaCy.
        
        Args:
            model_name: Название модели spaCy
            
        Returns:
            Загруженная модель spaCy
        """
        global spacy_models
        
        if model_name in spacy_models:
            return spacy_models[model_name]
        
        try:
            # Пробуем загрузить модель
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            
            # Кэшируем модель для будущего использования
            spacy_models[model_name] = nlp
            return nlp
        except Exception as e:
            logger.error(f"Error loading spaCy model '{model_name}': {str(e)}")
            logger.warning("Trying to download the model...")
            
            try:
                # Пробуем скачать модель
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                logger.info(f"Successfully downloaded and loaded spaCy model: {model_name}")
                
                # Кэшируем модель
                spacy_models[model_name] = nlp
                return nlp
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {str(download_error)}")
                logger.warning("Falling back to blank spaCy model")
                
                # Создаем blank модель как запасной вариант
                fallback_nlp = spacy.blank("ru")
                spacy_models[model_name] = fallback_nlp
                return fallback_nlp
    
    def _detect_content_type(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Определяет тип содержимого для выбора подходящего разделителя.
        
        Args:
            content: Текстовое содержимое документа
            metadata: Метаданные документа, могут содержать тип контента
            
        Returns:
            Тип содержимого ("markdown", "html", "code", "text")
        """
        # Если тип контента указан в метаданных, используем его
        if metadata and "content_type" in metadata:
            content_type = metadata["content_type"].lower()
            if content_type in CONTENT_TYPES:
                return content_type
        
        # Проверяем наличие markdown-заголовков
        if re.search(r'^#+\s+', content, re.MULTILINE):
            return "markdown"
        
        # Проверяем наличие HTML-тегов
        if re.search(r'<\/?[a-z][\s\S]*>', content):
            return "html"
        
        # Проверяем, похож ли текст на код
        code_indicators = [
            r'def\s+\w+\s*\(', r'class\s+\w+\s*\:', r'function\s+\w+\s*\(',
            r'if\s*\(.+\)\s*\{', r'for\s*\(.+\)\s*\{', r'while\s*\(.+\)\s*\{'
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, content):
                return "code"
        
        # По умолчанию считаем текстом
        return "text"
    
    def _split_with_langchain(self, content: str, content_type: str, metadata: Dict[str, Any]) -> List[LCDocument]:
        """
        Разбивает контент с помощью LangChain.
        
        Args:
            content: Текстовое содержимое
            content_type: Тип содержимого
            metadata: Метаданные документа
            
        Returns:
            Список фрагментов в формате LangChain
        """
        try:
            if content_type == "markdown":
                # Сначала разбиваем по заголовкам, затем по размеру чанков
                try:
                    md_docs = self.md_splitter.split_text(content)
                    return self.text_splitter.split_documents(md_docs)
                except Exception as e:
                    logger.warning(f"Failed to split markdown with headers: {str(e)}")
                    # Если не удалось разбить по заголовкам, используем обычный текстовый сплиттер
                    return self.text_splitter.create_documents([content], [metadata])
            
            elif content_type == "html":
                return self.html_splitter.create_documents([content], [metadata])
            
            elif content_type == "code":
                return self.code_splitter.create_documents([content], [metadata])
            
            else:  # text
                return self.text_splitter.create_documents([content], [metadata])
        except Exception as e:
            logger.error(f"LangChain splitting error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _split_with_spacy(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает текст на предложения с помощью spaCy.
        
        Args:
            content: Текстовое содержимое
            metadata: Метаданные документа
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        try:
            # Разбиваем текст на предложения
            sentences = self.nlp(content).sents
            
            chunks = []
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                # Если текущий chunk + новое предложение не превышает chunk_size,
                # добавляем предложение к текущему chunk
                if len(current_chunk) + len(sentence.text) <= self.chunk_size:
                    current_chunk += sentence.text + " "
                else:
                    # Если текущий chunk не пустой, сохраняем его
                    if current_chunk:
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = chunk_id
                        chunk_metadata["content_type"] = "text"
                        chunk_metadata["spacy_fallback"] = True
                        
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": chunk_metadata
                        })
                        
                        chunk_id += 1
                    
                    # Начинаем новый chunk
                    overlap_chars = min(len(current_chunk), self.chunk_overlap)
                    if overlap_chars > 0:
                        # Добавляем перекрытие
                        current_chunk = current_chunk[-overlap_chars:] + sentence.text + " "
                    else:
                        current_chunk = sentence.text + " "
            
            # Не забываем добавить последний chunk
            if current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["content_type"] = "text"
                chunk_metadata["spacy_fallback"] = True
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
            
            return chunks
        except Exception as e:
            logger.error(f"spaCy splitting error: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Умное разбиение текста на фрагменты с механизмами резервирования.
        
        Args:
            text: Исходный текст
            metadata: Метаданные для контекста
            
        Returns:
            Список словарей {text: str, metadata: dict}
        """
        metadata = metadata or {}
        content_type = self._detect_content_type(text, metadata)
        metadata["detected_content_type"] = content_type
        
        # Основной метод - LangChain
        try:
            # Получаем LangChain документы
            lc_docs = self._split_with_langchain(text, content_type, metadata)
            
            # Проверяем результат
            if not lc_docs:
                raise ValueError("LangChain splitting returned empty result")
            
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
            
        except Exception as e:
            logger.warning(f"Primary chunking method failed: {str(e)}")
            
            # Перебираем резервные методы
            for method in ["spacy"]:
                try:
                    logger.info(f"Trying fallback chunking method: {method}")
                    
                    if method == "spacy":
                        return self._split_with_spacy(text, metadata)
                    else:
                        logger.warning(f"Unknown fallback method: {method}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Fallback method {method} failed: {str(e)}")
                    continue
            
            # Если все методы не сработали, возвращаем один большой фрагмент
            logger.warning("All chunking methods failed. Returning a single chunk.")
            
            metadata["chunk_id"] = 0
            metadata["content_type"] = content_type
            metadata["emergency_fallback"] = True
            
            return [{
                "text": text,
                "metadata": metadata
            }]
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает фрагменты из документа с умным разбиением и резервными механизмами.
        
        Args:
            document: Документ для разбиения
            doc_id: Идентификатор документа
            
        Returns:
            Список фрагментов
        """
        # Подготавливаем метаданные документа
        doc_metadata = document.metadata.copy()
        doc_metadata["doc_id"] = doc_id
        
        # Умное разбиение с учетом структуры документа и резервными методами
        chunks_data = self.split_text(document.content, doc_metadata)
        
        # Преобразуем в формат Chunk
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            # Если chunk_id уже есть в метаданных, используем его
            chunk_id = chunk_data["metadata"].get("chunk_id", i)
            
            chunk = Chunk(
                text=chunk_data["text"],
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata=chunk_data["metadata"]
            )
            chunks.append(chunk)
        
        return chunks 