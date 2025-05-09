"""
Пакет для мониторинга производительности и качества RAG-системы.
"""

from .metrics import get_metrics_instance, RAGMetrics

__all__ = ['get_metrics_instance', 'RAGMetrics'] 