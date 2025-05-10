import os
import logging
import asyncio
import sys
from datetime import datetime
from dotenv import load_dotenv

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from app.logging_config import setup_logging
from app.database import create_tables
from app.ollama_client import get_ollama_instance
# Удалены неиспользуемые импорты
# from app.routers.auth import get_password_hash
# from sqlalchemy import text

# Импорт маршрутов
from app.routers import auth, rag, documents, model

# Настройка логирования через конфигурацию
setup_logging()
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Инициализация FastAPI приложения
app = FastAPI(title="RAG Service")

# Инициализация таблиц в базе данных при запуске приложения
create_tables()

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для продакшена нужно указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model_initialized = False

# Примечание: функция get_rag_service() перемещена в app/routers/rag.py
# для устранения циклических импортов

# Healthcheck эндпоинт
@app.get("/api/healthcheck")
async def healthcheck():
    """
    Проверка здоровья сервиса, возвращает информацию о статусе различных компонентов
    """
    try:
        # Проверяем доступность Ollama
        ollama_available = False
        ollama_model = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
        try:
            ollama = get_ollama_instance()
            ollama_available = await ollama.check_model_availability()
        except Exception as e:
            logger.warning(f"Ollama недоступен: {str(e)}")
        
        # Проверяем состояние службы RAG
        rag_service_available = False
        try:
            # Импортируем функцию get_rag_service из модуля routers/rag.py
            from app.routers.rag import get_rag_service
            # Пробуем получить службу RAG
            _ = get_rag_service()
            rag_service_available = True
        except Exception as e:
            logger.warning(f"Служба RAG недоступна: {str(e)}")
        
        # Собираем информацию о среде
        environment_info = {
            "embedding_model": os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
            "spacy_model": os.getenv("SPACY_MODEL", "ru_core_news_md"),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "400")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
            "python_version": os.getenv("PYTHON_VERSION", ""),
            "pytorch_version": torch.__version__ if 'torch' in globals() else "not available"
        }
        
        # Возвращаем полный отчет о состоянии
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "rag_service": "available" if rag_service_available else "unavailable",
                "ollama": "available" if ollama_available else "unavailable",
                "ollama_model": ollama_model
            },
            "environment": environment_info
        }
    except Exception as e:
        logger.error(f"Ошибка при проверке состояния: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


async def initialize_ollama():
    """Инициализирует клиент Ollama и пытается загрузить модель."""
    try:
        model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
        logger.info(f"Initializing Ollama client with model {model_name}")
        
        # Проверяем наличие Ollama
        ollama_client = get_ollama_instance(model_name)
        available = await ollama_client.check_model_availability()
        
        if not available:
            # Пытаемся автоматически запустить Ollama если она установлена
            import platform
            import subprocess
            
            system = platform.system()
            logger.info(f"Attempting to automatically start Ollama on {system}")
            
            try:
                if system == "Darwin":  # macOS
                    # Пробуем запустить через open
                    subprocess.Popen(["open", "-a", "Ollama"])
                    logger.info("Attempted to start Ollama app on macOS")
                elif system == "Windows":
                    # Пытаемся запустить исполняемый файл Ollama
                    subprocess.Popen(["ollama", "serve"])
                elif system == "Linux":
                    # Пытаемся запустить ollama serve
                    subprocess.Popen(["ollama", "serve"])
                
                # Ждем немного, чтобы Ollama успела запуститься
                logger.info("Waiting for Ollama to start...")
                await asyncio.sleep(5)
                
                # Проверяем снова доступность
                available = await ollama_client.check_model_availability()
                if available:
                    logger.info("Successfully started Ollama")
                else:
                    logger.warning("Could not automatically start Ollama")
            except Exception as e:
                logger.error(f"Error starting Ollama: {str(e)}")
        
        await ollama_client.ensure_model_loaded()
        return ollama_client
    except Exception as e:
        logger.error(f"Error initializing Ollama: {str(e)}")
        return None


# Событие при запуске приложения
@app.on_event("startup")
async def startup_event():
    """Выполняется при запуске приложения."""
    # Запускаем инициализацию модели в отдельном потоке
    from concurrent.futures import ThreadPoolExecutor
    
    executor = ThreadPoolExecutor()
    logger.info("Starting application...")
    
    # Инициализируем клиент Ollama асинхронно
    try:
        # Запускаем инициализацию в отдельном потоке
        executor.submit(initialize_ollama_sync)
        logger.info("Ollama initialization process started in background")
    except Exception as e:
        logger.error(f"Error starting Ollama initialization: {str(e)}")
    finally:
        executor.shutdown(wait=False)  # Закрываем executor без ожидания задач


def initialize_ollama_sync():
    """Синхронная обертка для запуска initialize_ollama в отдельном потоке."""
    try:
        # Создаем новый event loop для выполнения асинхронной функции
        asyncio.run(initialize_ollama())
    except Exception as e:
        logger.error(f"Error in Ollama initialization thread: {str(e)}")


# Подключение маршрутов
app.include_router(auth.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(model.router)
# Перенаправляем корневой маршрут /me на аутентификационный маршрут /me
app.add_api_route("/me", auth.router.routes[-1].endpoint, methods=["GET"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 