"""
Модуль для мониторинга производительности и качества RAG-системы.
"""

import time
import threading
import logging
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Deque
from collections import deque

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMetrics:
    """Класс для мониторинга производительности RAG-системы."""
    
    def __init__(self, metrics_dir: str = "app/monitoring/metrics"):
        """
        Инициализация системы метрик.
        
        Args:
            metrics_dir: Директория для сохранения метрик
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Метрики производительности
        self.latency_metrics: Dict[str, Deque[float]] = {
            "retrieval": deque(maxlen=1000),  # Последние 1000 значений
            "generation": deque(maxlen=1000),
            "total": deque(maxlen=1000)
        }
        
        # Метрики качества
        self.quality_metrics: Dict[str, Deque[float]] = {
            "answer_similarity": deque(maxlen=1000),
            "context_precision": deque(maxlen=1000),
            "context_recall": deque(maxlen=1000),
            "faithfulness": deque(maxlen=1000)
        }
        
        # Счетчики ошибок
        self.errors: Dict[str, int] = {
            "retrieval_errors": 0,
            "generation_errors": 0,
            "total_requests": 0
        }
        
        # История пользовательских запросов (для анализа дрифта)
        self.recent_queries: Deque[Dict[str, Any]] = deque(maxlen=100)
        
        # Мониторинг тепловой карты запросов
        self.query_heatmap: Dict[int, int] = {
            hour: 0 for hour in range(24)
        }
        
        # Запускаем периодическое сохранение метрик
        self.start_metrics_saver()
    
    def record_latency(self, phase: str, latency_ms: float):
        """
        Записывает метрику задержки.
        
        Args:
            phase: Фаза (retrieval, generation, total)
            latency_ms: Задержка в миллисекундах
        """
        if phase in self.latency_metrics:
            self.latency_metrics[phase].append(latency_ms)
    
    def record_quality(self, metric: str, value: float):
        """
        Записывает метрику качества.
        
        Args:
            metric: Название метрики
            value: Значение метрики
        """
        if metric in self.quality_metrics:
            self.quality_metrics[metric].append(value)
    
    def record_error(self, error_type: str):
        """
        Учитывает ошибку.
        
        Args:
            error_type: Тип ошибки
        """
        if error_type in self.errors:
            self.errors[error_type] += 1
        self.errors["total_requests"] += 1
    
    def record_query(self, query: str):
        """
        Записывает запрос пользователя.
        
        Args:
            query: Текст запроса
        """
        self.recent_queries.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query
        })
        
        # Обновляем тепловую карту запросов
        current_hour = datetime.datetime.now().hour
        self.query_heatmap[current_hour] += 1
        
        self.errors["total_requests"] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку метрик.
        
        Returns:
            Словарь с метриками
        """
        def safe_mean(values):
            return sum(values) / len(values) if values else 0
        
        # Готовим метрики производительности
        perf_metrics = {}
        for phase, values in self.latency_metrics.items():
            if values:
                perf_metrics[f"{phase}_avg_ms"] = safe_mean(values)
                perf_metrics[f"{phase}_p95_ms"] = self._percentile(values, 95)
                perf_metrics[f"{phase}_p99_ms"] = self._percentile(values, 99)
        
        # Готовим метрики качества
        quality_summary = {}
        for metric, values in self.quality_metrics.items():
            if values:
                quality_summary[f"{metric}_avg"] = safe_mean(values)
                quality_summary[f"{metric}_min"] = min(values)
        
        # Готовим статистику ошибок
        error_rate = 0
        if self.errors["total_requests"] > 0:
            error_rate = (self.errors["retrieval_errors"] + self.errors["generation_errors"]) / self.errors["total_requests"]
        
        return {
            "performance": perf_metrics,
            "quality": quality_summary,
            "error_rate": error_rate,
            "total_requests": self.errors["total_requests"],
            "query_distribution": dict(self.query_heatmap),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _percentile(self, values: List[float], p: int) -> float:
        """
        Рассчитывает перцентиль для списка значений.
        
        Args:
            values: Список значений
            p: Перцентиль (0-100)
            
        Returns:
            Значение перцентиля
        """
        if not values:
            return 0
            
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    def save_metrics(self):
        """Сохраняет текущие метрики в файл."""
        metrics = self.get_metrics_summary()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        metrics_file = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")
        
        try:
            # Загружаем существующие метрики, если есть
            existing_metrics = []
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    existing_metrics = json.load(f)
            
            # Добавляем новые метрики
            if isinstance(existing_metrics, list):
                existing_metrics.append(metrics)
            else:
                existing_metrics = [metrics]
            
            # Сохраняем
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def start_metrics_saver(self):
        """Запускает периодическое сохранение метрик."""
        def saver_thread():
            while True:
                try:
                    self.save_metrics()
                except Exception as e:
                    logger.error(f"Error in metrics saver thread: {str(e)}")
                # Сохраняем каждый час
                time.sleep(3600)
        
        threading.Thread(target=saver_thread, daemon=True).start()

# Глобальный экземпляр для метрик
rag_metrics_instance = None

def get_metrics_instance() -> RAGMetrics:
    """
    Получить глобальный экземпляр метрик.
    
    Returns:
        Экземпляр RAGMetrics
    """
    global rag_metrics_instance
    if rag_metrics_instance is None:
        rag_metrics_instance = RAGMetrics()
    return rag_metrics_instance 