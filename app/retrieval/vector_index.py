"""
Модуль с индексом векторов для обратной совместимости.
"""

# Импортируем QdrantIndex из основного модуля rag
from app.rag import QdrantIndex

# Переименовываем для обратной совместимости
VectorIndex = QdrantIndex 