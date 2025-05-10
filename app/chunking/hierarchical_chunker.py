"""
Модуль для иерархического разбиения документов на фрагменты.

Этот модуль позволяет разбивать структурированные документы (например, docx, pdf)
с сохранением иерархии заголовков и их отношения к тексту.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from app.chunking.unified_chunker import Document, Chunk, UnifiedChunker

# Настройка логирования
logger = logging.getLogger(__name__)

class HierarchicalChunker:
    """
    Чанкер, учитывающий иерархическую структуру документа.
    
    Позволяет разбивать документы, сохраняя информацию о заголовках,
    иерархии разделов и взаимосвязях между частями текста.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        heading_patterns: Optional[List[str]] = None,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000,
        max_tokens: int = 512,
        overlap_tokens: int = 20,
        language: str = "russian",
        spacy_model: str = "ru_core_news_md"
    ):
        """
        Инициализация иерархического чанкера.
        
        Args:
            chunk_size: Целевой размер чанка в символах
            chunk_overlap: Перекрытие между соседними чанками в символах
            heading_patterns: Регулярные выражения для определения заголовков разных уровней
            min_chunk_size: Минимальный размер чанка в символах
            max_chunk_size: Максимальный размер чанка в символах
            max_tokens: Максимальное количество токенов в чанке
            overlap_tokens: Перекрытие между токенами в чанке
            language: Язык текста
            spacy_model: Модель spaCy для обработки текста
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Паттерны для определения уровней заголовков
        self.heading_patterns = heading_patterns or [
            r"^#+\s+(.+)$",                  # Markdown заголовки (# Заголовок 1, ## Заголовок 2...)
            r"^Глава\s+\d+\.\s+(.+)$",       # Главы (Глава 1. Название)
            r"^Статья\s+\d+\.\s+(.+)$",      # Статьи (Статья 1. Название)
            r"^Раздел\s+[IVX]+\.\s+(.+)$",   # Разделы римскими цифрами (Раздел I. Название)
            r"^(\d+\.\d+)\.\s+(.+)$",        # Подразделы (1.1. Название)
            r"^(\d+\.\d+\.\d+)\.\s+(.+)$",   # Пункты (1.1.1. Название)
        ]
        
        # Дополнительные паттерны для распознавания частей документа
        self.section_patterns = {
            'definition': [r"определени[еяй]", r"термин[ыов]*"],
            'obligation': [r"обязан(ности|ных|а)", r"долж(ен|на|ны|но)"],
            'prohibition': [r"запрещ[ае][не][тоы]", r"не допуск[ае][ет]"],
            'penalty': [r"штраф", r"взыскани[еяй]", r"наказани[еяй]"],
            'procedure': [r"процедур[аы]", r"порядок", r"алгоритм"],
        }
        
        # Базовые паттерны для определения границ параграфов и предложений
        self.paragraph_sep = r"\n\s*\n"
        self.sentence_sep = r"[.!?][)\"]?\s+"
        
        logger.info(f"Initialized HierarchicalChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def _detect_heading_level(self, text: str) -> Tuple[int, Optional[str]]:
        """
        Определяет уровень заголовка и его текст.
        
        Args:
            text: Строка текста для анализа
            
        Returns:
            Кортеж (уровень заголовка, текст заголовка), где уровень 0 означает обычный текст
        """
        # Проверяем каждый паттерн по порядку
        for level, pattern in enumerate(self.heading_patterns, 1):
            match = re.match(pattern, text.strip())
            if match:
                # Получаем текст заголовка из группы захвата
                heading_text = match.group(1) if match.groups() else text.strip()
                return level, heading_text
        
        # Если не найдено совпадений, это не заголовок
        return 0, None
    
    def _classify_text_type(self, text: str) -> Dict[str, float]:
        """
        Классифицирует тип текста (определение, обязанность, запрет и т.д.)
        
        Args:
            text: Фрагмент текста для классификации
            
        Returns:
            Словарь с типами текста и их вероятностями
        """
        result = {}
        text_lower = text.lower()
        
        # Проверяем каждый тип по набору паттернов
        for text_type, patterns in self.section_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * 0.2  # Увеличиваем вес за каждое совпадение
            
            if score > 0:
                result[text_type] = min(score, 1.0)  # Нормализуем до 1.0
        
        return result
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Разбивает текст на параграфы по пустым строкам.
        
        Args:
            text: Текст для разбиения
            
        Returns:
            Список параграфов
        """
        return [p for p in re.split(self.paragraph_sep, text) if p.strip()]
    
    def _split_paragraph_by_size(self, paragraph: str, target_size: int) -> List[str]:
        """
        Разбивает длинный параграф на части подходящего размера по границам предложений.
        
        Args:
            paragraph: Текст параграфа
            target_size: Целевой размер фрагмента
            
        Returns:
            Список фрагментов параграфа
        """
        if len(paragraph) <= target_size:
            return [paragraph]
        
        parts = []
        sentences = re.split(self.sentence_sep, paragraph)
        current_part = ""
        
        for sentence in sentences:
            # Добавляем точку и пробел, если они были удалены при разделении
            if not sentence.endswith(('.', '!', '?')):
                sentence += '. '
            
            # Если текущая часть с новым предложением превышает целевой размер 
            # и текущая часть не пуста, сохраняем ее и начинаем новую
            if len(current_part + sentence) > target_size and current_part:
                parts.append(current_part.strip())
                current_part = sentence
            else:
                current_part += sentence
        
        # Добавляем последнюю часть
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    def _build_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """
        Строит иерархическую структуру документа.
        
        Args:
            text: Текст документа
            
        Returns:
            Список словарей, представляющих иерархию документа
        """
        paragraphs = self._split_into_paragraphs(text)
        
        hierarchy = []
        current_path = []  # Путь в иерархии [уровень1, уровень2, ...]
        content_buffer = ""
        
        for paragraph in paragraphs:
            level, heading_text = self._detect_heading_level(paragraph)
            
            # Если это заголовок
            if level > 0:
                # Сохраняем накопленный контент
                if content_buffer:
                    if current_path:
                        # Привязываем контент к последнему заголовку в иерархии
                        self._add_content_to_hierarchy(hierarchy, current_path, content_buffer)
                    else:
                        # Контент без заголовка в начале документа
                        hierarchy.append({
                            'level': 0,
                            'text': content_buffer,
                            'heading': None,
                            'children': []
                        })
                    content_buffer = ""
                
                # Обновляем текущий путь в иерархии
                while len(current_path) >= level:
                    current_path.pop()
                current_path.append(heading_text)
                
                # Добавляем новый заголовок в иерархию
                self._add_heading_to_hierarchy(hierarchy, current_path, level)
            else:
                # Добавляем параграф в буфер контента
                if content_buffer:
                    content_buffer += "\n\n"
                content_buffer += paragraph
        
        # Добавляем оставшийся контент
        if content_buffer:
            if current_path:
                self._add_content_to_hierarchy(hierarchy, current_path, content_buffer)
            else:
                hierarchy.append({
                    'level': 0,
                    'text': content_buffer,
                    'heading': None,
                    'children': []
                })
        
        return hierarchy
    
    def _add_heading_to_hierarchy(self, hierarchy: List[Dict[str, Any]], path: List[str], level: int):
        """Добавляет заголовок в иерархию документа."""
        if level == 1:
            # Заголовок верхнего уровня
            hierarchy.append({
                'level': level,
                'heading': path[0],
                'text': '',
                'children': []
            })
            return
        
        # Ищем родительский элемент в иерархии
        parent = self._find_node_in_hierarchy(hierarchy, path[:-1])
        if parent:
            parent['children'].append({
                'level': level,
                'heading': path[-1],
                'text': '',
                'children': []
            })
    
    def _add_content_to_hierarchy(self, hierarchy: List[Dict[str, Any]], path: List[str], content: str):
        """Добавляет контент к указанному заголовку в иерархии."""
        node = self._find_node_in_hierarchy(hierarchy, path)
        if node:
            if node['text']:
                node['text'] += "\n\n" + content
            else:
                node['text'] = content
    
    def _find_node_in_hierarchy(self, hierarchy: List[Dict[str, Any]], path: List[str]) -> Optional[Dict[str, Any]]:
        """Находит узел в иерархии по пути из заголовков."""
        if not path:
            return None
        
        # Ищем верхний уровень
        for item in hierarchy:
            if item.get('heading') == path[0]:
                # Если это конечный узел, возвращаем его
                if len(path) == 1:
                    return item
                
                # Иначе ищем дальше в дочерних элементах
                return self._find_node_in_hierarchy(item['children'], path[1:])
        
        return None
    
    def _hierarchy_to_chunks(self, hierarchy: List[Dict[str, Any]], doc_id: str) -> List[Chunk]:
        """
        Преобразует иерархическую структуру в список чанков с метаданными.
        
        Args:
            hierarchy: Иерархическая структура документа
            doc_id: Идентификатор документа
            
        Returns:
            Список объектов Chunk
        """
        chunks = []
        chunk_id = 0
        
        def process_node(node, parent_path=None):
            nonlocal chunk_id
            
            if parent_path is None:
                parent_path = []
            
            # Формируем путь к текущему узлу
            current_path = parent_path[:]
            if node.get('heading'):
                current_path.append(node['heading'])
            
            # Полный путь заголовков до текущего узла
            heading_path = " > ".join(current_path) if current_path else ""
            
            # Создаем метаданные чанка
            metadata = {
                'heading': node.get('heading', ''),
                'heading_path': heading_path,
                'level': node.get('level', 0),
                'section_type': {}
            }
            
            # Если у узла есть текст, создаем из него чанки
            if node.get('text'):
                text = node['text']
                
                # Определяем тип раздела
                section_types = self._classify_text_type(text)
                metadata['section_type'] = section_types
                
                # Разбиваем на параграфы
                paragraphs = self._split_into_paragraphs(text)
                
                # Объединяем параграфы в чанки нужного размера
                current_chunk = ""
                
                for paragraph in paragraphs:
                    # Если параграф слишком длинный, разбиваем его на части
                    if len(paragraph) > self.max_chunk_size:
                        parts = self._split_paragraph_by_size(paragraph, self.chunk_size)
                    else:
                        parts = [paragraph]
                    
                    for part in parts:
                        # Если добавление части превысит целевой размер чанка, сохраняем текущий чанк
                        if current_chunk and len(current_chunk + "\n\n" + part) > self.chunk_size:
                            # Если текущий чанк достаточно большой, создаем объект Chunk
                            if len(current_chunk) >= self.min_chunk_size:
                                chunks.append(Chunk(
                                    text=current_chunk,
                                    doc_id=doc_id,
                                    chunk_id=chunk_id,
                                    metadata=metadata.copy()
                                ))
                                chunk_id += 1
                            current_chunk = part
                        else:
                            # Добавляем часть к текущему чанку
                            if current_chunk:
                                current_chunk += "\n\n" + part
                            else:
                                current_chunk = part
                
                # Добавляем последний чанк
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=current_chunk,
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        metadata=metadata.copy()
                    ))
                    chunk_id += 1
            
            # Рекурсивно обрабатываем дочерние узлы
            for child in node.get('children', []):
                process_node(child, current_path)
        
        # Обрабатываем все узлы верхнего уровня
        for node in hierarchy:
            process_node(node)
        
        return chunks
    
    def create_chunks_from_document(self, document: Document, doc_id: str) -> List[Chunk]:
        """
        Создает чанки из документа с учетом иерархической структуры.
        
        Args:
            document: Документ для разбиения
            doc_id: Идентификатор документа
            
        Returns:
            Список объектов Chunk
        """
        logger.info(f"Creating hierarchical chunks for document {doc_id}")
        
        # Строим иерархическую структуру документа
        hierarchy = self._build_hierarchy(document.text)
        
        # Преобразуем структуру в чанки
        chunks = self._hierarchy_to_chunks(hierarchy, doc_id)
        
        # Добавляем метаданные документа к каждому чанку
        for chunk in chunks:
            chunk.metadata.update({
                'source': document.metadata.get('source', ''),
                'title': document.metadata.get('title', ''),
                'author': document.metadata.get('author', ''),
                'date': document.metadata.get('date', ''),
            })
        
        logger.info(f"Created {len(chunks)} hierarchical chunks for document {doc_id}")
        return chunks 