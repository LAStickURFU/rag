"""
Модуль конфигурации для RAG-системы.
Содержит настройки, загружаемые из переменных окружения с значениями по умолчанию.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from functools import lru_cache

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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.6"))
USE_ADAPTIVE_K = os.getenv("USE_ADAPTIVE_K", "true").lower() == "true"
LANGUAGE = os.getenv("LANGUAGE", "russian")
SPACY_MODEL = os.getenv("SPACY_MODEL", "ru_core_news_md")

# Настройки LLM
LLM_MODEL = os.getenv("LLM_MODEL", "mistral:7b-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# Настройки адаптивного количества фрагментов
MIN_K = int(os.getenv("MIN_K", "2"))
MAX_K = int(os.getenv("MAX_K", "5"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.45"))

# Настройки Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Сводная информация о конфигурации для отладки
# def get_config_summary():
#     ... # Не используется

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