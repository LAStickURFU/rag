"""
Модуль для сбора и хранения метрик производительности и качества RAG-системы.
"""

import logging
import time
from typing import Dict, Any, List, Optional
import threading

logger = logging.getLogger(__name__)

class MetricsSingleton:
    """
    Синглтон для сбора метрик по всему приложению.
    Обеспечивает запись метрик производительности, качества и ошибок.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsSingleton, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Инициализация хранилища метрик."""
        self.latency_metrics = {}
        self.error_counts = {}
        self.quality_metrics = {}
        self.recent_queries = []
        self.max_queries_stored = 100
        self.start_time = time.time()
    
    def record_latency(self, phase: str, elapsed_ms: float) -> None:
        """
        Записывает метрику времени выполнения.
        
        Args:
            phase: Название фазы (retrieval, generation, total)
            elapsed_ms: Время выполнения в миллисекундах
        """
        if phase not in self.latency_metrics:
            self.latency_metrics[phase] = {
                'count': 0,
                'total_ms': 0,
                'min_ms': float('inf'),
                'max_ms': 0,
                'recent': []
            }
        
        self.latency_metrics[phase]['count'] += 1
        self.latency_metrics[phase]['total_ms'] += elapsed_ms
        self.latency_metrics[phase]['min_ms'] = min(self.latency_metrics[phase]['min_ms'], elapsed_ms)
        self.latency_metrics[phase]['max_ms'] = max(self.latency_metrics[phase]['max_ms'], elapsed_ms)
        
        # Храним последние 10 значений для анализа тренда
        recent = self.latency_metrics[phase]['recent']
        recent.append(elapsed_ms)
        if len(recent) > 10:
            recent.pop(0)
    
    def record_error(self, error_type: str) -> None:
        """
        Записывает ошибку указанного типа.
        
        Args:
            error_type: Тип ошибки
        """
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def record_query(self, query: str) -> None:
        """
        Записывает запрос пользователя.
        
        Args:
            query: Запрос пользователя
        """
        # Обрезаем длинные запросы и сохраняем только последние N
        if len(query) > 200:
            query = query[:197] + "..."
        
        self.recent_queries.append({
            'query': query,
            'timestamp': time.time()
        })
        
        if len(self.recent_queries) > self.max_queries_stored:
            self.recent_queries.pop(0)
    
    def record_quality(self, metric_name: str, value: float) -> None:
        """
        Записывает метрику качества.
        
        Args:
            metric_name: Название метрики
            value: Значение метрики
        """
        if metric_name not in self.quality_metrics:
            self.quality_metrics[metric_name] = {
                'count': 0,
                'total': 0,
                'min': float('inf'),
                'max': 0,
                'recent': []
            }
        
        self.quality_metrics[metric_name]['count'] += 1
        self.quality_metrics[metric_name]['total'] += value
        self.quality_metrics[metric_name]['min'] = min(self.quality_metrics[metric_name]['min'], value)
        self.quality_metrics[metric_name]['max'] = max(self.quality_metrics[metric_name]['max'], value)
        
        # Храним последние 10 значений
        recent = self.quality_metrics[metric_name]['recent']
        recent.append(value)
        if len(recent) > 10:
            recent.pop(0)
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по времени выполнения.
        
        Returns:
            Словарь со статистикой по фазам
        """
        result = {}
        for phase, metrics in self.latency_metrics.items():
            if metrics['count'] > 0:
                avg = metrics['total_ms'] / metrics['count']
                result[phase] = {
                    'count': metrics['count'],
                    'avg_ms': round(avg, 2),
                    'min_ms': round(metrics['min_ms'], 2),
                    'max_ms': round(metrics['max_ms'], 2),
                    'recent': [round(x, 2) for x in metrics['recent']]
                }
        return result
    
    def get_error_stats(self) -> Dict[str, int]:
        """
        Возвращает статистику по ошибкам.
        
        Returns:
            Словарь с количеством ошибок по типам
        """
        return dict(self.error_counts)
    
    def get_quality_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику по метрикам качества.
        
        Returns:
            Словарь со статистикой по метрикам
        """
        result = {}
        for metric, stats in self.quality_metrics.items():
            if stats['count'] > 0:
                avg = stats['total'] / stats['count']
                result[metric] = {
                    'count': stats['count'],
                    'avg': round(avg, 4),
                    'min': round(stats['min'], 4),
                    'max': round(stats['max'], 4),
                    'recent': [round(x, 4) for x in stats['recent']]
                }
        return result
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Возвращает последние запросы.
        
        Args:
            limit: Максимальное число запросов
            
        Returns:
            Список последних запросов
        """
        return self.recent_queries[-limit:] if self.recent_queries else []
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает общую сводку метрик.
        
        Returns:
            Словарь со сводной информацией
        """
        uptime_seconds = time.time() - self.start_time
        
        total_queries = 0
        for phase, metrics in self.latency_metrics.items():
            if phase == 'total':
                total_queries = metrics['count']
                break
        
        return {
            'uptime_seconds': round(uptime_seconds, 2),
            'total_queries': total_queries,
            'error_rate': self._get_error_rate(),
            'avg_latency': self._get_avg_latency('total'),
            'performance_trend': self._get_performance_trend()
        }
    
    def _get_error_rate(self) -> float:
        """
        Вычисляет уровень ошибок.
        
        Returns:
            Процент ошибок от общего числа запросов
        """
        total_errors = sum(self.error_counts.values())
        total_queries = self.latency_metrics.get('total', {}).get('count', 0)
        
        if total_queries == 0:
            return 0.0
        
        return round((total_errors / total_queries) * 100, 2)
    
    def _get_avg_latency(self, phase: str) -> float:
        """
        Возвращает среднее время выполнения для указанной фазы.
        
        Args:
            phase: Название фазы
            
        Returns:
            Среднее время в миллисекундах
        """
        if phase not in self.latency_metrics or self.latency_metrics[phase]['count'] == 0:
            return 0.0
        
        return round(self.latency_metrics[phase]['total_ms'] / self.latency_metrics[phase]['count'], 2)
    
    def _get_performance_trend(self) -> str:
        """
        Анализирует тренд производительности.
        
        Returns:
            Строка с описанием тренда
        """
        if 'total' not in self.latency_metrics or len(self.latency_metrics['total']['recent']) < 5:
            return "Недостаточно данных"
        
        recent = self.latency_metrics['total']['recent']
        avg_first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        avg_second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        if avg_second_half < avg_first_half * 0.9:
            return "Улучшение"
        elif avg_second_half > avg_first_half * 1.1:
            return "Ухудшение"
        else:
            return "Стабильно"


def get_metrics_instance() -> MetricsSingleton:
    """
    Возвращает экземпляр синглтона для метрик.
    
    Returns:
        Экземпляр MetricsSingleton
    """
    return MetricsSingleton() 