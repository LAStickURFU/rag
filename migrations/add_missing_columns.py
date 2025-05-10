import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Column, Float, Integer, text
from dotenv import load_dotenv

# Добавляем путь к корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Загружаем переменные окружения
load_dotenv()

# Получаем URL базы данных из переменных окружения
DATABASE_URL = os.getenv("DATABASE_URL", 
                         "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres")


def run_migration():
    """Выполняет миграцию для добавления недостающих колонок."""
    try:
        # Создаем подключение к базе данных
        engine = create_engine(DATABASE_URL)
        
        # Создаем метаданные и загружаем существующую таблицу
        metadata = MetaData()
        metadata.reflect(bind=engine)
        
        # Проверяем наличие таблицы model_configs
        if 'model_configs' not in metadata.tables:
            print("Таблица model_configs не существует!")
            return
        
        # Получаем таблицу
        model_configs = metadata.tables['model_configs']
        
        # Создаем список колонок для добавления
        columns_to_add = []
        
        # Проверяем, есть ли колонка top_p
        if 'top_p' not in model_configs.columns:
            columns_to_add.append(('top_p', 'FLOAT', '0.9'))
            print("Колонка top_p будет добавлена")
            
        # Проверяем, есть ли колонка top_k_chunks
        if 'top_k_chunks' not in model_configs.columns:
            columns_to_add.append(('top_k_chunks', 'INTEGER', '5'))
            print("Колонка top_k_chunks будет добавлена")
            
        # Проверяем, есть ли колонка context_window
        if 'context_window' not in model_configs.columns:
            columns_to_add.append(('context_window', 'INTEGER', '8192'))
            print("Колонка context_window будет добавлена")
        
        # Если нет колонок для добавления, завершаем работу
        if not columns_to_add:
            print("Все необходимые колонки уже существуют!")
            return
        
        # Выполняем миграцию с помощью ALTER TABLE
        with engine.begin() as connection:
            for column_name, column_type, default_value in columns_to_add:
                sql = text(f"ALTER TABLE model_configs ADD COLUMN {column_name} {column_type} DEFAULT {default_value}")
                connection.execute(sql)
                print(f"Колонка {column_name} успешно добавлена")
        
        print("Миграция успешно завершена!")
    
    except Exception as e:
        print(f"Ошибка при выполнении миграции: {str(e)}")


if __name__ == "__main__":
    print(f"Начинаем миграцию: {datetime.now()}")
    run_migration()
    print(f"Миграция завершена: {datetime.now()}") 