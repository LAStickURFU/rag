"""
Модуль для реализации CrossEncoder для переранжирования результатов поиска.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Реализация CrossEncoder для более точного ранжирования результатов поиска.
    
    CrossEncoder принимает пару (запрос, документ) и выдает оценку релевантности.
    В отличие от bi-encoder (dense retrieval), CrossEncoder использует self-attention
    между запросом и документом, что обеспечивает более точные результаты.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Инициализация CrossEncoder.
        
        Args:
            model_name: Название модели из библиотеки Hugging Face
        """
        self.model_name = model_name
        
        # Загрузка модели и токенизатора
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Проверяем доступность GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            # Переводим модель в режим инференса
            self.model.eval()
            
            logger.info(f"Initialized CrossEncoder model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {str(e)}")
            # В случае ошибки создаем заглушку
            self.model = None
            self.tokenizer = None
    
    def is_available(self) -> bool:
        """Проверяет, успешно ли загружена модель."""
        return self.model is not None and self.tokenizer is not None
    
    def rerank(self, query: str, documents: List[str], scores: Optional[List[float]] = None) -> List[Tuple[int, float]]:
        """
        Переранжирование результатов поиска.
        
        Args:
            query: Текст запроса
            documents: Список текстов документов для ранжирования
            scores: Опциональный список начальных scores (например, от BM25)
            
        Returns:
            Список кортежей (индекс документа, score)
        """
        if not self.is_available():
            logger.warning("CrossEncoder model is not available, returning original order")
            
            # Если модель недоступна, возвращаем исходные scores или равные веса
            if scores is None:
                scores = [1.0 for _ in documents]
                
            return [(i, score) for i, score in enumerate(scores)]
        
        # Готовим пары (запрос, документ) для CrossEncoder
        pairs = [(query, doc) for doc in documents]
        
        try:
            # Токенизация с паддингом и трункацией
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Получаем предсказания модели
                outputs = self.model(**inputs)
                
                # Извлекаем scores
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits
                
                # Берем только score для положительного класса (релевантности)
                if logits.shape[1] > 1:  # Если модель имеет несколько выходов
                    scores = logits[:, 1].cpu().numpy()  # Берем вероятность релевантности
                else:
                    scores = logits.squeeze(-1).cpu().numpy()
                
                # Создаем список (индекс, score)
                ranked = [(i, float(score)) for i, score in enumerate(scores)]
                
                # Сортируем по убыванию score
                ranked.sort(key=lambda x: x[1], reverse=True)
                
                return ranked
                
        except Exception as e:
            logger.error(f"Error during CrossEncoder ranking: {str(e)}")
            
            # В случае ошибки возвращаем оригинальный порядок
            if scores is None:
                scores = [1.0 for _ in documents]
                
            return [(i, score) for i, score in enumerate(scores)]
    
    def rerank_with_original_weights(self, query: str, documents: List[str], 
                                    original_scores: List[float], 
                                    weight: float = 0.5) -> List[Tuple[int, float]]:
        """
        Переранжирование с учетом оригинальных весов и весов CrossEncoder.
        
        Args:
            query: Текст запроса
            documents: Список текстов документов для ранжирования
            original_scores: Список начальных scores (например, от BM25 или dense retrieval)
            weight: Вес CrossEncoder в финальном score (от 0 до 1)
            
        Returns:
            Список кортежей (индекс документа, score)
        """
        assert 0 <= weight <= 1, "Weight must be between 0 and 1"
        assert len(documents) == len(original_scores), "Number of documents and scores must match"
        
        # Получаем scores от CrossEncoder
        cross_encoder_results = self.rerank(query, documents)
        
        # Нормализуем CrossEncoder scores
        if cross_encoder_results:
            max_ce_score = max(score for _, score in cross_encoder_results)
            min_ce_score = min(score for _, score in cross_encoder_results)
            range_ce = max_ce_score - min_ce_score
            
            # Создаем словарь normalized CrossEncoder scores
            ce_scores = {}
            for idx, score in cross_encoder_results:
                if range_ce > 0:
                    ce_scores[idx] = (score - min_ce_score) / range_ce
                else:
                    ce_scores[idx] = 1.0
        else:
            ce_scores = {i: 1.0 for i in range(len(documents))}
        
        # Нормализуем original scores
        max_orig = max(original_scores)
        min_orig = min(original_scores)
        range_orig = max_orig - min_orig
        
        norm_orig_scores = {}
        for i, score in enumerate(original_scores):
            if range_orig > 0:
                norm_orig_scores[i] = (score - min_orig) / range_orig
            else:
                norm_orig_scores[i] = 1.0
        
        # Объединяем scores с заданными весами
        combined_scores = []
        for i in range(len(documents)):
            orig_score = norm_orig_scores[i]
            ce_score = ce_scores.get(i, 0.0)
            
            # Взвешенная сумма
            combined_score = (1 - weight) * orig_score + weight * ce_score
            combined_scores.append((i, combined_score))
        
        # Сортируем по убыванию
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores 