"""
Модуль для BM25 ретривера - компонент для sparse retrieval в гибридном поиске.
"""

import math
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter

import numpy as np
import spacy
from spacy.language import Language

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    Реализация BM25 алгоритма для sparse retrieval.
    
    BM25 - это функция ранжирования, использующая частотные характеристики 
    слов в документах для определения релевантности.
    """
    
    def __init__(self, 
                k1: float = 1.5, 
                b: float = 0.75, 
                use_lemmatization: bool = True,
                language: str = "russian",
                spacy_model: str = "ru_core_news_md"):
        """
        Инициализация BM25 ретривера.
        
        Args:
            k1: Параметр насыщения термина (обычно от 1.2 до 2.0)
            b: Параметр нормализации длины документа (обычно 0.75)
            use_lemmatization: Флаг использования лемматизации
            language: Язык для стоп-слов
            spacy_model: Модель spaCy для лемматизации
        """
        self.k1 = k1
        self.b = b
        self.use_lemmatization = use_lemmatization
        self.language = language
        self.spacy_model = spacy_model
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.warning(f"Could not load spaCy model {spacy_model}: {str(e)}. Using blank model.")
            self.nlp = spacy.blank("ru")
        self.stop_words = self.nlp.Defaults.stop_words
        
        # Будет инициализировано при индексации
        self.doc_freqs = {}  # Сколько документов содержат термин
        self.idf = {}  # Инвертированная частота документов
        self.doc_len = []  # Длина каждого документа
        self.avgdl = 0  # Средняя длина документа
        self.doc_tokens = []  # Токены в каждом документе
        self.total_docs = 0  # Общее количество документов
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Предобработка текста: лемматизация, удаление стоп-слов.
        
        Args:
            text: Входной текст
            
        Returns:
            Список токенов
        """
        doc = self.nlp(text.lower().strip())
        tokens = [token.lemma_ if self.use_lemmatization else token.text for token in doc
                  if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 1]
        return tokens
    
    def index_documents(self, texts: List[str]) -> None:
        """
        Индексация документов для BM25.
        
        Args:
            texts: Список текстов для индексации
        """
        self.doc_tokens = []
        self.doc_len = []
        self.doc_freqs = {}
        self.idf = {}
        self.total_docs = len(texts)
        
        # Индексируем каждый документ
        for text in texts:
            tokens = self.preprocess_text(text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            
            # Подсчитываем document frequency
            unique_tokens = set(tokens)
            for token in unique_tokens:
                if token not in self.doc_freqs:
                    self.doc_freqs[token] = 0
                self.doc_freqs[token] += 1
        
        # Вычисляем среднюю длину документа
        self.avgdl = sum(self.doc_len) / self.total_docs if self.total_docs > 0 else 0
        
        # Вычисляем IDF для каждого токена
        for token, freq in self.doc_freqs.items():
            # Сглаженная версия IDF
            self.idf[token] = math.log((self.total_docs - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def _score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        """
        Вычисляет BM25 score для документа.
        
        Args:
            query_tokens: Токены запроса
            doc_idx: Индекс документа
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        # Счетчик токенов в документе
        doc_token_counts = Counter(doc_tokens)
        
        # Нормализация длины документа
        len_norm = (1 - self.b) + self.b * (doc_len / self.avgdl) if self.avgdl > 0 else 1.0
        
        # Для каждого токена запроса
        for token in query_tokens:
            if token not in self.idf:
                continue
                
            # TF термина в документе
            term_freq = doc_token_counts.get(token, 0)
            
            # BM25 формула
            numerator = self.idf[token] * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * len_norm
            
            score += numerator / denominator if denominator != 0 else 0
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество возвращаемых документов
            
        Returns:
            Список кортежей (индекс документа, score)
        """
        query_tokens = self.preprocess_text(query)
        
        # Вычисляем score для каждого документа
        scores = [(i, self._score_doc(query_tokens, i)) for i in range(self.total_docs)]
        
        # Сортируем по убыванию score и берем top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k] 