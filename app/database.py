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
    """Создание таблиц в базе данных и применение всех необходимых миграций."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Проверяем и добавляем все необходимые колонки,
        # которые могли бы быть добавлены через миграции
        apply_all_schema_updates()
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise


def apply_all_schema_updates():
    """
    Применяет все необходимые обновления схемы базы данных.
    Эта функция объединяет все миграции в одно обновление.
    """
    try:
        with engine.connect() as conn:
            # 1. Убедимся, что таблица users имеет колонку role
            update_users_table(conn)
            
            # 2. Убедимся, что таблица model_configs имеет колонку model_name с правильным значением по умолчанию
            update_model_configs_table(conn)
            
            # 3. Убедимся, что таблица documents имеет все необходимые колонки для метаданных
            update_documents_table(conn)
            
            # 4. Убедимся, что таблица chats имеет колонку relevant_chunks
            update_chats_table(conn)
            
            # Фиксируем все изменения
            conn.commit()
            
        logger.info("All schema updates applied successfully")
    except Exception as e:
        logger.error(f"Error applying schema updates: {str(e)}")
        raise


def update_users_table(conn):
    """Обновляет таблицу users."""
    # Список колонок для проверки и добавления
    columns_to_check = [
        ("role", "VARCHAR", "'user'"),
        ("email", "VARCHAR", "NULL"),
        ("is_active", "BOOLEAN", "true"),
        ("updated_at", "TIMESTAMP", "NOW()")
    ]
    
    for column_name, column_type, default_value in columns_to_check:
        result = conn.execute(text(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name='users' AND column_name='{column_name}'"
        ))
        if result.rowcount == 0:
            conn.execute(text(
                f"ALTER TABLE users ADD COLUMN {column_name} {column_type} "
                f"DEFAULT {default_value}"
            ))
            logger.info(f"Added missing column '{column_name}' to users table")
    
    # Создаем индекс для email
    try:
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email)"
        ))
        logger.info("Created index for email in users table")
    except Exception as e:
        logger.warning(f"Could not create index for email: {str(e)}")


def update_model_configs_table(conn):
    """Обновляет таблицу model_configs."""
    # Список проверок для таблицы model_configs
    columns_to_check = [
        # Колонка, тип, значение по умолчанию
        ("model_name", "VARCHAR", "'mistral:7b-instruct'"),
        ("top_p", "FLOAT", "0.9"),
        ("top_k_chunks", "INTEGER", "5"),
        ("context_window", "INTEGER", "8192"),
        ("use_query_processing", "BOOLEAN", "false"),
        ("use_query_rephrase", "BOOLEAN", "false"),
        ("use_chain_of_thought", "BOOLEAN", "false")
    ]
    
    for column_name, column_type, default_value in columns_to_check:
        result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name='model_configs' AND column_name='{column_name}'"))
        if result.rowcount == 0:
            conn.execute(text(f"ALTER TABLE model_configs ADD COLUMN {column_name} {column_type} DEFAULT {default_value}"))
            logger.info(f"Added missing column '{column_name}' to model_configs table")
    
    # Обновляем значения model_name, где они равны 'mistral' или NULL
    conn.execute(text("UPDATE model_configs SET model_name = 'mistral:7b-instruct' WHERE model_name = 'mistral' OR model_name IS NULL"))
    logger.info("Updated model_name values in model_configs table")


def update_documents_table(conn):
    """Обновляет таблицу documents, добавляя колонки для метаданных."""
    # Список колонок для проверки и добавления
    columns_to_check = [
        ("chunk_size", "INTEGER"),
        ("chunk_overlap", "INTEGER"),
        ("embedding_model", "VARCHAR"),
        ("processing_params", "JSONB"),
        ("chunking_mode", "VARCHAR")
    ]
    
    for column_name, column_type in columns_to_check:
        result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name='documents' AND column_name='{column_name}'"))
        if result.rowcount == 0:
            default_value = "'character'" if column_name == "chunking_mode" else ""
            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {column_name} {column_type} {default_value}"))
            logger.info(f"Added missing column '{column_name}' to documents table")
    
    # Устанавливаем значение по умолчанию для chunking_mode, если оно отсутствует
    conn.execute(text("UPDATE documents SET chunking_mode = 'character' WHERE chunking_mode IS NULL"))


def update_chats_table(conn):
    """Обновляет таблицу chats, добавляя колонку relevant_chunks."""
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='chats' AND column_name='relevant_chunks'"))
    if result.rowcount == 0:
        conn.execute(text("ALTER TABLE chats ADD COLUMN relevant_chunks JSON DEFAULT '[]'"))
        logger.info("Added missing column 'relevant_chunks' to chats table")


def get_db():
    """Зависимость для получения сессии базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 