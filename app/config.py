"""
Модуль конфигурации для RAG-системы.
Содержит настройки, загружаемые из переменных окружения с значениями по умолчанию.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from functools import lru_cache
from dataclasses import dataclass

# Загружаем переменные окружения из .env файла
load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = os.getenv("INDEX_DIR", str(DATA_DIR / "indexes"))

# Настройки базы данных
# Формат: postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres")

# Настройки Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Настройки RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "7"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.6"))
USE_ADAPTIVE_K = os.getenv("USE_ADAPTIVE_K", "true").lower() == "true"
LANGUAGE = os.getenv("LANGUAGE", "russian")
SPACY_MODEL = os.getenv("SPACY_MODEL", "ru_core_news_md")

# Настройки LLM
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Настройки адаптивного количества фрагментов
MIN_K = int(os.getenv("MIN_K", "2"))
MAX_K = int(os.getenv("MAX_K", "5"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.45"))

# Настройки Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Настройки токенового чанкера
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "64"))
USE_TOKEN_CHUNKER = os.getenv("USE_TOKEN_CHUNKER", "false").lower() == "true"
TOKEN_MODEL = os.getenv("TOKEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Режим чанкинга: character, token, semantic, hierarchical
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "character")

# Сводная информация о конфигурации для отладки
# def get_config_summary():
#     ... # Не используется

@dataclass
class QdrantConfig:
    """Конфигурация для Qdrant."""
    
    host: str = QDRANT_HOST
    port: int = QDRANT_PORT
    
    @property
    def url(self) -> str:
        """URL для подключения к Qdrant."""
        return f"http://{self.host}:{self.port}"


class RagConfig:
    """Конфигурация системы RAG."""
    
    def __init__(self):
        """Инициализация конфигурации RAG."""
        self.model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        self.use_hybrid = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
        self.use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
        self.use_adaptive_k = os.getenv("USE_ADAPTIVE_K", "true").lower() == "true"
        self.dense_weight = float(os.getenv("DENSE_WEIGHT", "0.6"))
        self.sparse_weight = float(os.getenv("SPARSE_WEIGHT", "0.4"))
        self.reranker_weight = float(os.getenv("RERANKER_WEIGHT", "0.6"))
        self.max_context_docs = int(os.getenv("MAX_CONTEXT_DOCS", "5"))
        self.top_k_chunks = int(os.getenv("TOP_K_CHUNKS", "7"))
        
        # Параметры символьного чанкера (для обратной совместимости)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
        
        # Параметры токенового чанкера
        self.use_token_chunker = os.getenv("USE_TOKEN_CHUNKER", "false").lower() == "true"
        self.max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        self.overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "64"))
        self.token_model_name = os.getenv("TOKEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        
        # Параметры языка
        self.language = os.getenv("LANGUAGE", "russian")
        self.spacy_model = os.getenv("SPACY_MODEL", "ru_core_news_md")
        
        # Режим чанкинга
        self.chunking_mode = os.getenv("CHUNKING_MODE", "character")
        if self.use_token_chunker:
            # Для обратной совместимости
            self.chunking_mode = "token"
            
        # Настройки для иерархического чанкинга
        self.heading_patterns = os.getenv("HEADING_PATTERNS", "").split('|') if os.getenv("HEADING_PATTERNS") else None
        self.min_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "50"))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
        
        # Параметры для пакетной обработки
        self.batch_size = int(os.getenv("BATCH_SIZE", "64"))

@lru_cache()
def get_config() -> Dict[str, Any]:
    """
    Получение конфигурации приложения.
    Возвращает кэшированный словарь с параметрами.
    """
    return {
        # Настройки для эмбеддингов и индексирования
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "CROSS_ENCODER_MODEL": CROSS_ENCODER_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K_CHUNKS": TOP_K_CHUNKS,
        
        # Настройки для гибридного поиска
        "USE_HYBRID_SEARCH": USE_HYBRID_SEARCH,
        "USE_RERANKER": USE_RERANKER,
        "DENSE_WEIGHT": DENSE_WEIGHT,
        "RERANKER_WEIGHT": RERANKER_WEIGHT,
        
        # Настройки для LLM
        "LLM_MODEL": LLM_MODEL,
        "LLM_TEMPERATURE": LLM_TEMPERATURE,
        "LLM_MAX_TOKENS": LLM_MAX_TOKENS,
        
        # Настройки адаптивного количества фрагментов
        "USE_ADAPTIVE_K": USE_ADAPTIVE_K,
        "MIN_K": MIN_K,
        "MAX_K": MAX_K,
        "MIN_SCORE_THRESHOLD": MIN_SCORE_THRESHOLD,
        
        # Настройки Qdrant
        "QDRANT_HOST": QDRANT_HOST,
        "QDRANT_PORT": QDRANT_PORT,
    }

def get_embedding_model() -> str:
    """Получение названия модели для эмбеддингов."""
    return get_config()["EMBEDDING_MODEL"]

def get_llm_model() -> str:
    """Получение названия модели для LLM."""
    return get_config()["LLM_MODEL"]

def get_chunk_size() -> int:
    """Получение размера чанка."""
    return get_config()["CHUNK_SIZE"]

def get_chunk_overlap() -> int:
    """Получение перекрытия чанков."""
    return get_config()["CHUNK_OVERLAP"]

def get_top_k_chunks() -> int:
    """Получение количества чанков для контекста."""
    return get_config()["TOP_K_CHUNKS"]

def create_rag_service_from_config(custom_config=None):
    """
    Фабричный метод для создания экземпляра RAGService с согласованными параметрами из конфигурации.
    
    Args:
        custom_config: Опциональная пользовательская конфигурация, которая переопределяет настройки
                     из переменных окружения
                     
    Returns:
        Экземпляр RAGService с согласованными параметрами
    """
    from app.rag import RAGService
    
    # Получаем базовую конфигурацию
    config = custom_config or RagConfig()
    
    # Создаем экземпляр RAGService с параметрами из конфигурации
    rag_service = RAGService(
        model_name=config.model_name,
        collection_name=config.collection_name,
        use_hybrid=config.use_hybrid,
        use_reranker=config.use_reranker,
        use_adaptive_k=config.use_adaptive_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
        reranker_weight=config.reranker_weight,
        max_context_docs=config.max_context_docs,
        top_k_chunks=config.top_k_chunks,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        use_token_chunker=config.chunking_mode == "token",
        max_tokens=config.max_tokens,
        overlap_tokens=config.overlap_tokens,
        chunking_mode=config.chunking_mode,
        min_chunk_size=config.min_chunk_size,
        max_chunk_size=config.max_chunk_size,
        heading_patterns=config.heading_patterns,
        language=config.language,
        spacy_model=config.spacy_model
    )
    
    return rag_service