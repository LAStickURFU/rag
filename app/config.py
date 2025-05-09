"""
Модуль конфигурации для RAG-системы.
Содержит настройки, загружаемые из переменных окружения с значениями по умолчанию.
"""

import os
from pathlib import Path

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
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "True").lower() in ["true", "1", "yes"]
USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() in ["true", "1", "yes"]
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.5"))
USE_ADAPTIVE_K = os.getenv("USE_ADAPTIVE_K", "True").lower() in ["true", "1", "yes"]
LANGUAGE = os.getenv("LANGUAGE", "russian")
SPACY_MODEL = os.getenv("SPACY_MODEL", "ru_core_news_md")

# Сводная информация о конфигурации для отладки
# def get_config_summary():
#     ... # Не используется