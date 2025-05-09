"""
Модуль token-based чанкера для более эффективного разбиения документов.
Использует токенизатор трансформерной модели вместо разбиения по символам.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from transformers import AutoTokenizer
import numpy as np

# Импорт из проекта
from app.chunking.robust_chunker import Document, Chunk, CONTENT_TYPES

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenChunker:
    """
    Класс для разбиения документов на чанки с использованием токенизатора модели.
    Это позволяет создавать более эффективные чанки, оптимизированные под размер 
    контекстного окна модели.
    """
    
    def __init__(self, 
                max_tokens: int = 512, 
                overlap_tokens: int = 20,
                model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                language: str = "russian"):
        """
        Инициализация токенового чанкера.
        
        Args:
            max_tokens: Максимальное количество токенов в чанке
            overlap_tokens: Перекрытие между соседними чанками в токенах
            model_name: Название модели для токенизатора
            language: Язык документов для языкоспецифичных методов разбиения
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.language = language
        self.model_name = model_name
        
        # Загружаем токенизатор
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer from {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {model_name}: {str(e)}")
            # Используем запасной токенизатор, если основной не загрузился
            try:
                fallback_model = "intfloat/multilingual-e5-large"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                logger.info(f"Loaded fallback tokenizer from {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback tokenizer: {str(fallback_error)}")
                raise RuntimeError("Could not load any tokenizer")
    
    def chunk_text_by_tokens(self, text: str) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки на основе токенов.
        
        Args:
            text: Текст для разбиения
            
        Returns:
            Список словарей {text: str, token_count: int}
        """
        # Токенизируем текст
        tokens = self.tokenizer.encode(text)
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
            
            # Сохраняем чанк
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "metadata": {
                    "chunk_id": chunk_id,
                    "content_type": "text",
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end
                }
            })
            
            # Переходим к следующему чанку с учетом перекрытия
            start += self.max_tokens - self.overlap_tokens
            chunk_id += 1
        
        return chunks
    
    def _chunk_semantic_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Разбивает текст на семантические параграфы, а затем на чанки с учетом токенов.
        
        Args:
            text: Текст для разбиения
            
        Returns:
            Список словарей {text: str, token_count: int, metadata: dict}
        """
        # Разбиваем на параграфы
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_tokens = []
        current_paragraphs = []
        chunk_id = 0
        
        for paragraph in paragraphs:
            # Токенизируем параграф
            para_tokens = self.tokenizer.encode(paragraph)
            
            # Если параграф слишком большой, разбиваем его отдельно
            if len(para_tokens) > self.max_tokens:
                # Сначала сохраняем накопленные параграфы, если они есть
                if current_tokens:
                    combined_text = "\n\n".join(current_paragraphs)
                    chunks.append({
                        "text": combined_text,
                        "token_count": len(current_tokens),
                        "metadata": {
                            "chunk_id": chunk_id,
                            "content_type": "text",
                            "token_count": len(current_tokens)
                        }
                    })
                    chunk_id += 1
                    current_tokens = []
                    current_paragraphs = []
                
                # Затем разбиваем большой параграф токенами
                para_chunks = self.chunk_text_by_tokens(paragraph)
                for chunk in para_chunks:
                    chunk["metadata"]["chunk_id"] = chunk_id
                    chunks.append(chunk)
                    chunk_id += 1
            
            # Если текущий параграф помещается в чанк, добавляем его
            elif len(current_tokens) + len(para_tokens) <= self.max_tokens:
                current_tokens.extend(para_tokens)
                current_paragraphs.append(paragraph)
            
            # Иначе сохраняем текущий чанк и начинаем новый с этим параграфом
            else:
                combined_text = "\n\n".join(current_paragraphs)
                chunks.append({
                    "text": combined_text,
                    "token_count": len(current_tokens),
                    "metadata": {
                        "chunk_id": chunk_id,
                        "content_type": "text",
                        "token_count": len(current_tokens)
                    }
                })
                chunk_id += 1
                current_tokens = para_tokens
                current_paragraphs = [paragraph]
        
        # Сохраняем последний чанк, если есть накопленные параграфы
        if current_paragraphs:
            combined_text = "\n\n".join(current_paragraphs)
            chunks.append({
                "text": combined_text,
                "token_count": len(current_tokens),
                "metadata": {
                    "chunk_id": chunk_id,
                    "content_type": "text",
                    "token_count": len(current_tokens)
                }
            })
        
        return chunks
    
    def _detect_content_type(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Определяет тип содержимого для выбора подходящего метода разбиения.
        
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
        
        # По умолчанию считаем текстом
        return "text"
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Разбивает текст на чанки с учетом типа контента.
        
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
        
        # Для простого текста используем семантическое разбиение на параграфы
        if content_type == "text" or content_type == "markdown":
            return self._chunk_semantic_paragraphs(text)
        
        # Для других типов контента пока используем простое токеновое разбиение
        # В будущем можно добавить специализированные методы для разных типов
        return self.chunk_text_by_tokens(text)
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает фрагменты из документа с использованием токенового разбиения.
        
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
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            # Если chunk_id уже есть в метаданных, используем его
            chunk_id = chunk_data["metadata"].get("chunk_id", i)
            
            chunk = Chunk(
                text=chunk_data["text"],
                doc_id=doc_id,
                chunk_id=chunk_id,
                metadata=chunk_data.get("metadata", {})
            )
            chunks.append(chunk)
        
        return chunks


def get_default_token_chunker() -> TokenChunker:
    """
    Создает и возвращает экземпляр TokenChunker с оптимизированными параметрами.
    
    Returns:
        Экземпляр TokenChunker
    """
    # 512 токенов - примерно 25% от контекстного окна в 2048 токенов
    # 20 токенов перекрытия - около 4% от размера чанка, обеспечивает непрерывность контекста
    return TokenChunker(
        max_tokens=512,
        overlap_tokens=20,
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    ) 