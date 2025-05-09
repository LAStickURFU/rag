"""
Модуль для настройки и работы с базой данных.
"""

from sqlalchemy import create_engine
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


def get_db():
    """Зависимость для получения сессии базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 