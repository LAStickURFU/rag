"""
Модуль для улучшенного разбиения документов на фрагменты с механизмами резервирования.
"""

import re
import json
import csv
import io
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
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
CONTENT_TYPES = ["text", "markdown", "html", "pdf", "code", "json", "csv", "xml", "yaml", "txt"]

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
        
        # JSON сплиттер с учетом структуры JSON
        self.json_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "},{", "},\n{", "}\n{", 
                "],[", "],\n[", "]\n[",
                "\n\n", "\n", ". ", " ", ""
            ],
            keep_separator=True
        )
        
        # CSV сплиттер с учетом строк
        self.csv_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n", ",", " ", ""
            ],
            keep_separator=True
        )
        
        # XML сплиттер с учетом XML-тегов
        self.xml_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "</", ">", "\n", " ", ""
            ],
            keep_separator=True
        )
        
        # YAML сплиттер
        self.yaml_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n---", "\n...", "\n", ": ", " ", ""
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
            Тип содержимого ("markdown", "html", "code", "text", "json", "csv", "xml", "yaml")
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
        content_sample = content.strip()[:1000]  # Проверяем первые 1000 символов для экономии ресурсов
        
        # Проверка на JSON
        if content_sample.startswith('{') and content_sample.endswith('}') or \
           content_sample.startswith('[') and content_sample.endswith(']'):
            try:
                json.loads(content_sample if len(content_sample) < 1000 else content_sample + ']' 
                          if content_sample.startswith('[') else content_sample + '}')
                return "json"
            except json.JSONDecodeError:
                pass  # Не JSON, продолжаем проверки
        
        # Проверка на CSV
        if ',' in content_sample and '\n' in content_sample:
            try:
                csv_reader = csv.reader(io.StringIO(content_sample))
                # Проверяем, что первые несколько строк имеют одинаковое количество полей
                rows = list(csv_reader)
                if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows) and len(rows[0]) > 1:
                    return "csv"
            except Exception:
                pass  # Не CSV, продолжаем проверки
        
        # Проверка на XML
        if content_sample.startswith('<?xml') or \
           (content_sample.startswith('<') and '>' in content_sample and '</' in content_sample):
            return "xml"
        
        # Проверка на YAML
        if re.search(r'^[\w\s]+:\s+[\w\s]+$', content_sample, re.MULTILINE) and \
           not re.search(r'<\/?[a-z][\s\S]*>', content_sample):
            return "yaml"
        
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
                
            elif content_type == "json":
                # Для JSON используем специализированный метод разбиения
                json_chunks = self._split_json(content, metadata)
                # Преобразуем результат в формат LangChain
                return [LCDocument(page_content=chunk["text"], metadata=chunk["metadata"]) 
                        for chunk in json_chunks]
                
            elif content_type == "csv":
                return self.csv_splitter.create_documents([content], [metadata])
                
            elif content_type == "xml":
                return self.xml_splitter.create_documents([content], [metadata])
                
            elif content_type == "yaml":
                return self.yaml_splitter.create_documents([content], [metadata])
            
            else:  # text и другие форматы
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
        
        # Используем специализированные методы разбиения для разных типов контента
        try:
            # Выбираем метод разбиения в зависимости от типа контента
            if content_type == "json":
                return self._split_json(text, metadata)
            elif content_type == "csv":
                return self._split_csv(text, metadata)
            elif content_type in ["yaml", "xml", "txt"]:
                return self._split_text_with_metadata(text, metadata, content_type)
            else:
                # Для остальных типов используем основной метод LangChain
                lc_docs = self._split_with_langchain(text, content_type, metadata)
                
                # Проверяем результат
                if not lc_docs:
                    raise ValueError(f"LangChain splitting returned empty result for {content_type}")
                
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
    
    def _split_json(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает JSON на осмысленные фрагменты.
        
        Args:
            content: JSON-контент в виде строки
            metadata: Метаданные документа
            
        Returns:
            Список словарей {text: str, metadata: dict} с разбитым JSON
        """
        chunks = []
        chunk_id = 0
        
        try:
            # Попытка разобрать JSON
            json_data = json.loads(content)
            
            # Если это список объектов, разбиваем по элементам списка
            if isinstance(json_data, list):
                current_chunk = []
                current_size = 0
                
                for i, item in enumerate(json_data):
                    # Преобразуем элемент в строку
                    item_str = json.dumps(item, ensure_ascii=False, indent=2)
                    item_size = len(item_str)
                    
                    # Если элемент слишком большой, разбиваем его отдельно
                    if item_size > self.chunk_size:
                        # Если есть накопленные элементы, сохраняем их как чанк
                        if current_chunk:
                            chunk_text = "[\n" + ",\n".join([json.dumps(item, ensure_ascii=False, indent=2) 
                                                            for item in current_chunk]) + "\n]"
                            
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = chunk_id
                            chunk_metadata["json_items"] = f"items {i-len(current_chunk)}-{i-1}"
                            chunk_metadata["content_type"] = "json"
                            
                            chunks.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                            
                            chunk_id += 1
                            current_chunk = []
                            current_size = 0
                        
                        # Разбиваем большой элемент на части
                        sub_chunks = self._split_large_json_item(item, metadata, i)
                        for sub_chunk in sub_chunks:
                            sub_chunk["metadata"]["chunk_id"] = chunk_id
                            chunks.append(sub_chunk)
                            chunk_id += 1
                    
                    # Если добавление текущего элемента превысит размер чанка, создаем новый чанк
                    elif current_size + item_size > self.chunk_size and current_chunk:
                        chunk_text = "[\n" + ",\n".join([json.dumps(item, ensure_ascii=False, indent=2) 
                                                        for item in current_chunk]) + "\n]"
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = chunk_id
                        chunk_metadata["json_items"] = f"items {i-len(current_chunk)}-{i-1}"
                        chunk_metadata["content_type"] = "json"
                        
                        chunks.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })
                        
                        chunk_id += 1
                        current_chunk = [item]
                        current_size = item_size
                    
                    # Иначе добавляем элемент к текущему чанку
                    else:
                        current_chunk.append(item)
                        current_size += item_size
                
                # Не забываем про последний чанк
                if current_chunk:
                    chunk_text = "[\n" + ",\n".join([json.dumps(item, ensure_ascii=False, indent=2) 
                                                    for item in current_chunk]) + "\n]"
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["json_items"] = f"items {len(json_data)-len(current_chunk)}-{len(json_data)-1}"
                    chunk_metadata["content_type"] = "json"
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
            
            # Если это объект, разбиваем по ключам
            elif isinstance(json_data, dict):
                # Группируем ключи в чанки
                keys = list(json_data.keys())
                current_keys = []
                current_size = 0
                
                for i, key in enumerate(keys):
                    # Получаем значение по ключу и оцениваем его размер
                    value = json_data[key]
                    key_value_str = f'"{key}": {json.dumps(value, ensure_ascii=False, indent=2)}'
                    key_value_size = len(key_value_str)
                    
                    # Если значение слишком большое, обрабатываем его отдельно
                    if key_value_size > self.chunk_size:
                        # Если есть накопленные ключи, сохраняем их как чанк
                        if current_keys:
                            sub_dict = {k: json_data[k] for k in current_keys}
                            chunk_text = json.dumps(sub_dict, ensure_ascii=False, indent=2)
                            
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = chunk_id
                            chunk_metadata["json_keys"] = ", ".join(current_keys)
                            chunk_metadata["content_type"] = "json"
                            
                            chunks.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                            
                            chunk_id += 1
                            current_keys = []
                            current_size = 0
                        
                        # Обрабатываем большое значение
                        sub_chunks = self._split_large_json_value(key, value, metadata)
                        for sub_chunk in sub_chunks:
                            sub_chunk["metadata"]["chunk_id"] = chunk_id
                            chunks.append(sub_chunk)
                            chunk_id += 1
                    
                    # Если добавление текущего ключа превысит размер чанка, создаем новый чанк
                    elif current_size + key_value_size > self.chunk_size and current_keys:
                        sub_dict = {k: json_data[k] for k in current_keys}
                        chunk_text = json.dumps(sub_dict, ensure_ascii=False, indent=2)
                        
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = chunk_id
                        chunk_metadata["json_keys"] = ", ".join(current_keys)
                        chunk_metadata["content_type"] = "json"
                        
                        chunks.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })
                        
                        chunk_id += 1
                        current_keys = [key]
                        current_size = key_value_size
                    
                    # Иначе добавляем ключ к текущему чанку
                    else:
                        current_keys.append(key)
                        current_size += key_value_size
                
                # Не забываем про последний чанк
                if current_keys:
                    sub_dict = {k: json_data[k] for k in current_keys}
                    chunk_text = json.dumps(sub_dict, ensure_ascii=False, indent=2)
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["json_keys"] = ", ".join(current_keys)
                    chunk_metadata["content_type"] = "json"
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
            
            # Если после разбиения не получилось чанков, возвращаем весь JSON как один чанк
            if not chunks:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = 0
                chunk_metadata["content_type"] = "json"
                chunk_metadata["json_full"] = True
                
                chunks.append({
                    "text": content,
                    "metadata": chunk_metadata
                })
            
            return chunks
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error: {str(e)}. Falling back to standard text splitting.")
            # Если не удалось разобрать JSON, используем обычный текстовый сплиттер
            return self._split_with_langchain(content, "text", metadata)
    
    def _split_large_json_item(self, item: Any, metadata: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        """
        Разбивает большой элемент JSON-списка на части.
        
        Args:
            item: Элемент JSON-списка
            metadata: Метаданные документа
            index: Индекс элемента в списке
            
        Returns:
            Список словарей {text: str, metadata: dict} с разбитым элементом
        """
        chunks = []
        
        # Преобразуем элемент в строку
        item_str = json.dumps(item, ensure_ascii=False, indent=2)
        
        # Используем обычный текстовый сплиттер для разбиения
        lc_chunks = self.json_splitter.split_text(item_str)
        
        for i, chunk_text in enumerate(lc_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["json_item_index"] = index
            chunk_metadata["json_item_part"] = f"{i+1}/{len(lc_chunks)}"
            chunk_metadata["content_type"] = "json"
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _split_large_json_value(self, key: str, value: Any, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает большое значение JSON-объекта на части.
        
        Args:
            key: Ключ в JSON-объекте
            value: Значение, которое нужно разбить
            metadata: Метаданные документа
            
        Returns:
            Список словарей {text: str, metadata: dict} с разбитым значением
        """
        chunks = []
        
        # Преобразуем значение в строку
        value_str = json.dumps(value, ensure_ascii=False, indent=2)
        
        # Используем обычный текстовый сплиттер для разбиения
        lc_chunks = self.json_splitter.split_text(value_str)
        
        for i, chunk_text in enumerate(lc_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["json_key"] = key
            chunk_metadata["json_value_part"] = f"{i+1}/{len(lc_chunks)}"
            chunk_metadata["content_type"] = "json"
            
            # Форматируем текст как фрагмент JSON-объекта
            formatted_text = f'{{\n  "{key}" (part {i+1}/{len(lc_chunks)}): {chunk_text}\n}}'
            
            chunks.append({
                "text": formatted_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _split_csv(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Разбивает CSV на группы строк.
        
        Args:
            content: CSV-контент в виде строки
            metadata: Метаданные документа
            
        Returns:
            Список словарей {text: str, metadata: dict} с разбитым CSV
        """
        chunks = []
        chunk_id = 0
        
        try:
            # Создаем CSV-ридер
            csv_reader = csv.reader(io.StringIO(content))
            rows = list(csv_reader)
            
            if not rows:
                return self._split_with_langchain(content, "text", metadata)
            
            # Получаем заголовки (первая строка) и данные
            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            
            # Если CSV слишком маленький, возвращаем его целиком
            if len(data_rows) < 10:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["content_type"] = "csv"
                chunk_metadata["csv_rows"] = f"all ({len(data_rows)})"
                
                chunks.append({
                    "text": content,
                    "metadata": chunk_metadata
                })
                return chunks
            
            # Разбиваем на чанки
            current_rows = [headers]  # Всегда включаем заголовки
            current_size = len(','.join(headers))
            
            for i, row in enumerate(data_rows):
                row_size = len(','.join(row))
                
                # Если добавление строки превышает размер чанка, создаем новый чанк
                if current_size + row_size > self.chunk_size and len(current_rows) > 1:
                    # Создаем CSV из текущих строк
                    output = io.StringIO()
                    csv_writer = csv.writer(output)
                    csv_writer.writerows(current_rows)
                    chunk_text = output.getvalue()
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_id"] = chunk_id
                    chunk_metadata["content_type"] = "csv"
                    chunk_metadata["csv_rows"] = f"{i-len(current_rows)+2}-{i+1}"
                    
                    chunks.append({
                        "text": chunk_text,
                        "metadata": chunk_metadata
                    })
                    
                    chunk_id += 1
                    current_rows = [headers, row]  # Начинаем с заголовков и текущей строки
                    current_size = len(','.join(headers)) + row_size
                else:
                    current_rows.append(row)
                    current_size += row_size
            
            # Не забываем про последний чанк
            if len(current_rows) > 1:
                output = io.StringIO()
                csv_writer = csv.writer(output)
                csv_writer.writerows(current_rows)
                chunk_text = output.getvalue()
                
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["content_type"] = "csv"
                chunk_metadata["csv_rows"] = f"{len(data_rows)-len(current_rows)+2}-{len(data_rows)+1}"
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            return chunks
            
        except Exception as e:
            logger.warning(f"CSV splitting error: {str(e)}. Falling back to standard text splitting.")
            # Если не удалось разобрать CSV, используем обычный текстовый сплиттер
            return self._split_with_langchain(content, "text", metadata)
            
    def _split_text_with_metadata(self, content: str, metadata: Dict[str, Any], content_type: str) -> List[Dict[str, Any]]:
        """
        Разбивает простой текст с добавлением информации о типе контента в метаданные.
        
        Args:
            content: Текстовое содержимое
            metadata: Метаданные документа
            content_type: Тип контента (txt, yaml и т.д.)
            
        Returns:
            Список словарей {text: str, metadata: dict} с разбитым текстом
        """
        # Используем соответствующий сплиттер в зависимости от типа
        if content_type == "yaml":
            splitter = self.yaml_splitter
        elif content_type == "xml":
            splitter = self.xml_splitter
        else:  # txt и другие
            splitter = self.text_splitter
            
        try:
            # Разбиваем текст
            lc_chunks = splitter.split_text(content)
            
            # Преобразуем в нужный формат
            chunks = []
            for i, chunk_text in enumerate(lc_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["content_type"] = content_type
                
                if content_type == "yaml":
                    chunk_metadata["yaml_part"] = f"{i+1}/{len(lc_chunks)}"
                elif content_type == "xml":
                    chunk_metadata["xml_part"] = f"{i+1}/{len(lc_chunks)}"
                else:
                    chunk_metadata["text_part"] = f"{i+1}/{len(lc_chunks)}"
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            return chunks
        except Exception as e:
            logger.warning(f"{content_type.upper()} splitting error: {str(e)}. Falling back to standard text splitting.")
            # Если произошла ошибка, возвращаем весь контент в одном чанке
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = 0
            chunk_metadata["content_type"] = content_type
            chunk_metadata["fallback"] = True
            
            return [{
                "text": content,
                "metadata": chunk_metadata
            }] 