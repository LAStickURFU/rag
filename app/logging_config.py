"""
Конфигурация логирования для RAG системы.
"""

import os
import logging
import logging.config
from pathlib import Path

# Создаем директорию для логов, если ее нет
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Основные файлы логов
DEFAULT_LOG_FILE = os.getenv('LOG_FILE_PATH', 'logs/rag_backend.log')
OLLAMA_LOG_FILE = os.getenv('OLLAMA_LOG_FILE_PATH', 'logs/ollama.log')
RETRIEVAL_LOG_FILE = os.getenv('RETRIEVAL_LOG_FILE_PATH', 'logs/retrieval.log')
RAG_LOG_FILE = os.getenv('RAG_LOG_FILE_PATH', 'logs/rag.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Конфигурация логирования
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': LOG_LEVEL,
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': DEFAULT_LOG_FILE,
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'ollama_file': {
            'level': LOG_LEVEL,
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': OLLAMA_LOG_FILE,
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 3,
            'encoding': 'utf8'
        },
        'retrieval_file': {
            'level': LOG_LEVEL,
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': RETRIEVAL_LOG_FILE,
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 3,
            'encoding': 'utf8'
        },
        'rag_file': {
            'level': LOG_LEVEL,
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': RAG_LOG_FILE,
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 3,
            'encoding': 'utf8'
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        },
        'app.ollama_client': {
            'handlers': ['console', 'ollama_file'],
            'level': LOG_LEVEL,
            'propagate': False
        },
        'app.rag': {
            'handlers': ['console', 'rag_file'],
            'level': LOG_LEVEL,
            'propagate': False
        },
        'app.retrieval': {
            'handlers': ['console', 'retrieval_file'],
            'level': LOG_LEVEL,
            'propagate': False
        },
        'uvicorn': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.access': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

def setup_logging():
    """Настраивает логгер согласно конфигурации."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено. Уровень логов: {LOG_LEVEL}, основной файл: {DEFAULT_LOG_FILE}") 