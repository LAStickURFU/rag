"""
Модуль для настройки и работы с базой данных.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from app.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = DATABASE_URL
if not SQLALCHEMY_DATABASE_URL.startswith('postgresql'):
    raise RuntimeError(
        'Только PostgreSQL поддерживается. Проверь переменную окружения DATABASE_URL.'
    )
db_info = SQLALCHEMY_DATABASE_URL.split('@')[-1]
logger.info(f"Using database: {db_info}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def create_tables():
    """Создание таблиц в базе данных."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise


def update_schema():
    """
    Проверяет и обновляет схему базы данных.
    Добавляет отсутствующие колонки в существующие таблицы.
    """
    try:
        conn = engine.connect()
        # Проверяем наличие колонки role в таблице users
        result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='role'"))
        if result.rowcount == 0:
            # Колонка role отсутствует, добавляем её
            conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'"))
            logger.info("Added missing column 'role' to users table")
        conn.close()
    except Exception as e:
        logger.error(f"Error updating database schema: {str(e)}")
        raise


def get_db():
    """Зависимость для получения сессии базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 