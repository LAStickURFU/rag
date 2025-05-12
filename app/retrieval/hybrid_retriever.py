"""
Модуль для гибридного поиска, объединяющий dense и sparse retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.cross_encoder_reranker import CrossEncoderReranker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Гибридный поисковик, объединяющий результаты dense и sparse retrieval.
    
    Реализует стратегии:
    1. Взвешивание и слияние результатов
    2. Использование CrossEncoder для переранжирования
    3. Адаптивный выбор количества документов
    """
    
    def __init__(self, 
                dense_weight: float = 0.6,  # Оптимизировано для e5-base
                reranker_weight: float = 0.6,  # Усилено влияние reranker для более точного ранжирования
                use_reranker: bool = True,
                use_adaptive_k: bool = True,
                cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                language: str = "russian",
                spacy_model: str = "ru_core_news_md"):
        """
        Инициализация гибридного поисковика.
        
        Args:
            dense_weight: Вес dense retrieval (0-1)
            reranker_weight: Вес reranker в итоговом ранжировании (0-1)
            use_reranker: Использовать ли переранжирование
            use_adaptive_k: Использовать ли адаптивное количество документов
            cross_encoder_model: Название модели CrossEncoder
            language: Язык для BM25
            spacy_model: Название модели spaCy
        """
        self.dense_weight = dense_weight
        self.reranker_weight = reranker_weight
        self.use_reranker = use_reranker
        self.use_adaptive_k = use_adaptive_k
        self.language = language
        self.spacy_model = spacy_model
        
        # Инициализируем BM25 на spaCy
        self.bm25 = BM25Retriever(language=language, spacy_model=spacy_model)
        
        # Инициализируем переранжирование, если требуется
        if use_reranker:
            try:
                self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
                logger.info(f"Initialized CrossEncoder reranker: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize CrossEncoder: {str(e)}")
                self.reranker = None
                self.use_reranker = False
        else:
            self.reranker = None
    
    def index_documents(self, chunks: List[Any]) -> None:
        """
        Индексирует документы для BM25 (dense retrieval индексирование происходит в ChromaDB).
        
        Args:
            chunks: Список фрагментов с текстовым содержимым
        """
        # Извлекаем тексты из chunks
        texts = [chunk.text for chunk in chunks]
        
        # Индексируем для BM25
        self.bm25.index_documents(texts)
        
        logger.info(f"Indexed {len(texts)} documents for BM25 retrieval")
    
    def _normalize_scores(self, scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Нормализует оценки (scores) результатов.
        
        Args:
            scores: Список кортежей (индекс, оценка)
            
        Returns:
            Список кортежей (индекс, нормализованная оценка)
        """
        if not scores:
            return []
            
        # Извлекаем оценки
        indices = [idx for idx, _ in scores]
        values = [score for _, score in scores]
        
        # Находим min и max
        min_val = min(values)
        max_val = max(values)
        
        # Проверка на случай, если все значения равны
        if max_val == min_val:
            # Если все значения одинаковы, возвращаем единичную нормализацию
            return [(idx, 1.0) for idx in indices]
        
        # Нормализуем в диапазон [0, 1]
        normalized = [(idx, (val - min_val) / (max_val - min_val)) 
                     for idx, val in zip(indices, values)]
                     
        return normalized
    
    def _adaptive_k(self, scores: List[Tuple[int, float]], min_k: int = 5, max_k: int = 10) -> int:
        """
        Адаптивно выбирает количество документов на основе распределения scores.
        
        Args:
            scores: Список (индекс, score)
            min_k: Минимальное количество документов
            max_k: Максимальное количество документов
            
        Returns:
            Адаптивное количество документов
        """
        if not scores:
            return min_k
            
        # Сортируем по убыванию score
        sorted_scores = sorted([score for _, score in scores], reverse=True)
        
        if len(sorted_scores) <= min_k:
            return len(sorted_scores)
        
        # Вычисляем относительное падение score
        relative_drops = []
        for i in range(1, len(sorted_scores)):
            if sorted_scores[i-1] > 0:
                relative_drop = (sorted_scores[i-1] - sorted_scores[i]) / sorted_scores[i-1]
                relative_drops.append((i, relative_drop))
        
        # Если нет падений или слишком мало результатов, возвращаем минимум
        if not relative_drops:
            return min(min_k, len(sorted_scores))
        
        # Находим значительные падения после min_k
        significant_drops = [(i, drop) for i, drop in relative_drops if drop > 0.1 and i >= min_k]
        
        if significant_drops:
            # Берем первое значительное падение
            return significant_drops[0][0]
        else:
            # Если нет значительных падений, смотрим, какой минимальный score приемлем
            # Берем все документы со score не менее 70% от максимального (увеличено для лучшего качества ответов)
            threshold = sorted_scores[0] * 0.7
            for i, score in enumerate(sorted_scores):
                if score < threshold:
                    return max(min_k, i)
                    
            # Если все документы выше порога, берем все (не больше max_k)
            return min(len(sorted_scores), max_k)
    
    def _rrf_score(self, rank: int, k: int = 60) -> float:
        """
        Вычисляет RRF (Reciprocal Rank Fusion) оценку для ранга.
        
        Формула: 1 / (k + rank), где k - константа (обычно = 60).
        
        Args:
            rank: Ранг документа (начиная с 1)
            k: Константа RRF (по умолчанию 60 по рекомендации исследований)
            
        Returns:
            RRF оценка
        """
        return 1.0 / (k + rank)
    
    def search(self, query: str, dense_results: List[Tuple[Any, float]], 
              top_k: int = 10) -> List[Tuple[Any, float]]:
        """
        Выполняет гибридный поиск, объединяя dense и sparse результаты.
        
        Args:
            query: Поисковый запрос
            dense_results: Результаты dense retrieval (список кортежей (chunk, score))
            top_k: Количество возвращаемых результатов
            
        Returns:
            Список кортежей (chunk, score) с переранжированными результатами
        """
        chunks = [chunk for chunk, _ in dense_results]
        dense_scores = [score for _, score in dense_results]
        
        # Получаем результаты BM25 с увеличенным количеством кандидатов
        bm25_results = self.bm25.search(query, top_k=top_k*2)
        
        # Нормализуем scores
        norm_dense_scores = self._normalize_scores([(i, score) for i, score in enumerate(dense_scores)])
        norm_bm25_scores = self._normalize_scores(bm25_results)
        
        # Преобразуем нормализованные scores в ранги для Reciprocal Rank Fusion
        dense_ranks = {}
        bm25_ranks = {}
        
        # Сортируем по убыванию score и присваиваем ранги
        sorted_dense = sorted(norm_dense_scores, key=lambda x: x[1], reverse=True)
        sorted_bm25 = sorted(norm_bm25_scores, key=lambda x: x[1], reverse=True)
        
        for i, (idx, _) in enumerate(sorted_dense):
            dense_ranks[idx] = i + 1  # Ранги начинаются с 1
            
        for i, (idx, _) in enumerate(sorted_bm25):
            bm25_ranks[idx] = i + 1  # Ранги начинаются с 1
        
        # Объединяем результаты с использованием Reciprocal Rank Fusion
        hybrid_scores = []
        for i, chunk in enumerate(chunks):
            # Получаем ранги (если документ не найден, используем максимальный ранг + 1)
            dense_rank = dense_ranks.get(i, len(chunks) + 1)
            bm25_rank = bm25_ranks.get(i, len(chunks) + 1)
            
            # Вычисляем RRF scores
            rrf_dense = self._rrf_score(dense_rank)
            rrf_bm25 = self._rrf_score(bm25_rank)
            
            # Взвешенная сумма RRF scores
            hybrid_score = self.dense_weight * rrf_dense + (1 - self.dense_weight) * rrf_bm25
            hybrid_scores.append((i, hybrid_score))
        
        # Сортируем по убыванию score
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Добавляем Dynamic Rerank Threshold
        MIN_RERANK_THRESHOLD = 0.25  # Понижен с 0.3 для более широкого охвата кандидатов
        
        # Отбираем все фрагменты выше порога для переранжирования
        threshold_scores = [(idx, score) for idx, score in hybrid_scores 
                           if score >= MIN_RERANK_THRESHOLD]
        
        # Если нет результатов выше порога, берем top_k
        if not threshold_scores:
            threshold_scores = hybrid_scores[:top_k]
            
        # Ограничиваем количество для производительности
        max_rerank_candidates = min(top_k * 4, len(threshold_scores))  # Увеличено с top_k * 3 для более широкого охвата
        rerank_candidates = threshold_scores[:max_rerank_candidates]
        
        # Формируем данные для переранжирования
        rerank_indices = [idx for idx, _ in rerank_candidates]
        rerank_chunks = [chunks[idx] for idx in rerank_indices]
        rerank_chunk_texts = [chunk.text for chunk in rerank_chunks]
        rerank_scores = [score for _, score in rerank_candidates]
        
        logger.info(f"Selected {len(rerank_candidates)} candidates for reranking (min threshold: {MIN_RERANK_THRESHOLD})")
        
        # Если используем переранжирование и reranker доступен
        final_results = []
        if self.use_reranker and self.reranker and self.reranker.is_available():
            try:
                # Анализируем разброс scores в предварительных результатах
                score_range = max(rerank_scores) - min(rerank_scores) if rerank_scores else 0
                
                # Если разброс высокий, значит есть четкое разделение по релевантности
                # и CrossEncoder может быть более полезен
                dynamic_weight = min(0.8, self.reranker_weight + (score_range * 0.3))
                logger.info(f"Using dynamic reranker weight: {dynamic_weight:.2f}")
                
                # Переранжируем с учетом оригинальных scores и динамического веса
                reranker_results = self.reranker.rerank_with_original_weights(
                    query, rerank_chunk_texts, rerank_scores, weight=dynamic_weight
                )
                
                # Формируем окончательные результаты
                for idx, score in reranker_results:
                    orig_idx = rerank_indices[idx]
                    final_results.append((chunks[orig_idx], score))
                    
                logger.info(f"Reranked {len(reranker_results)} results using CrossEncoder")
                
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}")
                # Если переранжирование не удалось, возвращаем гибридные результаты
                for idx, score in rerank_candidates:
                    final_results.append((chunks[idx], score))
        else:
            # Если переранжирование не используется, возвращаем гибридные результаты
            for idx, score in rerank_candidates:
                final_results.append((chunks[idx], score))
        
        # Отсортировать финальные результаты по убыванию score и ограничить top_k
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Определяем адаптивное количество документов, если нужно
        if self.use_adaptive_k:
            adaptive_k = self._adaptive_k([(i, score) for i, (_, score) in enumerate(final_results)], 
                                        min_k=7, max_k=top_k)
            logger.info(f"Adaptive K selected: {adaptive_k} (from max top_k: {top_k})")
            top_k = adaptive_k
        
        # Возвращаем финальные результаты с ограничением top_k
        return final_results[:top_k] 