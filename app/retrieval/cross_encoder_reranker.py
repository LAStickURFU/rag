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
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):  # L-12 вместо L-6
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
            
            # Оптимизация для инференса
            if torch.cuda.is_available():
                # Используем оптимизации для ускорения инференса
                if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    self.use_amp = True
                    logger.info("Используем mixed precision для CrossEncoder")
                else:
                    self.use_amp = False
            else:
                self.use_amp = False
            
            logger.info(f"Initialized CrossEncoder model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {str(e)}")
            # В случае ошибки создаем заглушку
            self.model = None
            self.tokenizer = None
            self.use_amp = False
    
    def is_available(self) -> bool:
        """Проверяет, успешно ли загружена модель."""
        return self.model is not None and self.tokenizer is not None
    
    def rerank(self, query: str, documents: List[str], scores: Optional[List[float]] = None, 
              batch_size: int = 8) -> List[Tuple[int, float]]:
        """
        Переранжирование результатов поиска.
        
        Args:
            query: Текст запроса
            documents: Список текстов документов для ранжирования
            scores: Опциональный список начальных scores (например, от BM25)
            batch_size: Размер батча для обработки
            
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
            # Инициализация результатов
            all_scores = []
            
            # Обработка по батчам для более эффективного использования GPU
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Токенизация с паддингом и трункацией
                inputs = self.tokenizer(
                    batch_pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Получаем предсказания модели с поддержкой mixed precision
                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                
                # Извлекаем scores
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits
                
                # Берем только score для положительного класса (релевантности)
                if logits.shape[1] > 1:  # Если модель имеет несколько выходов
                    batch_scores = logits[:, 1].cpu().numpy()  # Берем вероятность релевантности
                else:
                    batch_scores = logits.squeeze(-1).cpu().numpy()
                
                # Добавляем в общий список
                all_scores.extend(batch_scores)
            
            # Создаем список (индекс, score)
            ranked = [(i, float(score)) for i, score in enumerate(all_scores)]
            
            # Сортируем по убыванию score
            ranked.sort(key=lambda x: x[1], reverse=True)
            
            # Логируем информацию о ранжировании для отладки
            if len(ranked) > 0:
                max_score = max(score for _, score in ranked)
                min_score = min(score for _, score in ranked)
                avg_score = sum(score for _, score in ranked) / len(ranked)
                logger.debug(f"CrossEncoder scores - Max: {max_score:.4f}, Min: {min_score:.4f}, Avg: {avg_score:.4f}")
            
            return ranked
        
        except Exception as e:
            logger.error(f"Error during CrossEncoder ranking: {str(e)}")
            
            # В случае ошибки возвращаем оригинальный порядок
            if scores is None:
                scores = [1.0 for _ in documents]
                
            return [(i, score) for i, score in enumerate(scores)]
    
    def rerank_with_original_weights(self, query: str, documents: List[str], 
                                    original_scores: List[float], 
                                    weight: float = 0.5,
                                    batch_size: int = 8) -> List[Tuple[int, float]]:
        """
        Переранжирование с учетом оригинальных весов и весов CrossEncoder.
        
        Args:
            query: Текст запроса
            documents: Список текстов документов для ранжирования
            original_scores: Список начальных scores (например, от BM25 или dense retrieval)
            weight: Вес CrossEncoder в финальном score (от 0 до 1)
            batch_size: Размер батча для обработки
            
        Returns:
            Список кортежей (индекс документа, score)
        """
        assert 0 <= weight <= 1, "Weight must be between 0 and 1"
        assert len(documents) == len(original_scores), "Number of documents and scores must match"
        
        # Получаем scores от CrossEncoder
        cross_encoder_results = self.rerank(query, documents, batch_size=batch_size)
        
        # Нормализуем CrossEncoder scores с более агрессивным нелинейным преобразованием
        if cross_encoder_results:
            max_ce_score = max(score for _, score in cross_encoder_results)
            min_ce_score = min(score for _, score in cross_encoder_results)
            range_ce = max_ce_score - min_ce_score
            
            # Создаем словарь normalized CrossEncoder scores
            ce_scores = {}
            for idx, score in cross_encoder_results:
                if range_ce > 0:
                    # Применяем сигмоидную трансформацию для усиления контраста
                    normalized_score = (score - min_ce_score) / range_ce
                    # Используем более сильную нелинейность для усиления контраста
                    enhanced_score = normalized_score ** 4 if normalized_score > 0.8 else (normalized_score ** 3 if normalized_score > 0.6 else normalized_score ** 2)
                    # Отсекаем очень низкие скоры
                    ce_scores[idx] = max(0.0, enhanced_score - 0.1)
                else:
                    ce_scores[idx] = 1.0
                
                # Логируем сырые и нормализованные скоры для отладки
                logger.debug(f"Doc {idx} CE raw score: {score:.4f}, normalized: {ce_scores[idx]:.4f}")
        else:
            ce_scores = {i: 1.0 for i in range(len(documents))}
        
        # Нормализуем original scores с более агрессивным отсечением
        max_orig = max(original_scores)
        min_orig = min(original_scores)
        range_orig = max_orig - min_orig
        
        norm_orig_scores = {}
        for i, score in enumerate(original_scores):
            if range_orig > 0:
                orig_norm = (score - min_orig) / range_orig
                # Применяем квадратичную функцию и отсечение низких значений
                norm_orig_scores[i] = max(0.0, orig_norm ** 2 - 0.1)
            else:
                norm_orig_scores[i] = 1.0
        
        # Объединяем scores с заданными весами, добавляя больше нелинейности
        combined_scores = []
        for i in range(len(documents)):
            orig_score = norm_orig_scores[i]
            ce_score = ce_scores.get(i, 0.0)
            
            # Применяем взвешенное произведение вместо суммы для более резкого контраста
            # Это даст высокий результат только если оба метода дали высокий скор
            combined_score = (orig_score ** (1 - weight)) * (ce_score ** weight)
            
            # Дополнительный бонус для документов с высоким score по обоим метрикам
            if orig_score > 0.7 and ce_score > 0.7:
                combined_score += 0.2  # Повышенный бонус для очень уверенных совпадений
            
            # Отсекаем низкие значения для более выраженного контраста
            final_score = max(0.0, combined_score)
            
            combined_scores.append((i, final_score))
            
            # Логируем детали ранжирования
            logger.debug(f"Doc {i}: orig={orig_score:.4f}, ce={ce_score:.4f}, combined={final_score:.4f}")
        
        # Сортируем по убыванию
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Логируем итоговые рейтинги
        for i, (idx, score) in enumerate(combined_scores[:5]):
            logger.info(f"Rank {i+1}: Doc {idx} with score {score:.4f}")
        
        return combined_scores 