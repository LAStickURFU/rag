"""
Пакет для оценки и анализа RAG-системы.

Этот пакет содержит модули и методы для оценки эффективности компонентов RAG-системы,
включая метрики retrieval и answer generation, а также логирование и анализ ошибок.
"""

import os

# Создаем необходимые директории
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "error_logs"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "datasets"), exist_ok=True)

from .logger import RAGEvaluationLogger

__all__ = ['RAGEvaluationLogger'] 