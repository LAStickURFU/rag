"""
Модуль chunking содержит компоненты для разбиения документов на фрагменты.
"""

from app.chunking.unified_chunker import (
    Document, 
    Chunk, 
    UnifiedChunker, 
    get_chunker,
    CONTENT_TYPES
)

from app.chunking.hierarchical_chunker import HierarchicalChunker

__all__ = [
    "Document",
    "Chunk",
    "UnifiedChunker",
    "HierarchicalChunker",
    "get_chunker",
    "CONTENT_TYPES"
]