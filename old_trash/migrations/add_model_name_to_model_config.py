import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, MetaData, text
from dotenv import load_dotenv

# Добавляем путь к корневой директории проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Загружаем переменные окружения
load_dotenv()

# Получаем URL базы данных из переменных окружения
DATABASE_URL = os.getenv("DATABASE_URL", 
                         "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres")

def run_migration():
    """Выполняет миграцию для добавления поля model_name в таблицу model_configs"""
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
        
        # Проверяем, есть ли уже поле model_name в таблице
        if 'model_name' in model_configs.columns:
            print("Поле model_name уже существует в таблице model_configs")
            # Обновляем значение по умолчанию для существующих записей
            with engine.begin() as connection:
                sql = text("UPDATE model_configs SET model_name = 'mistral:7b-instruct' WHERE model_name = 'mistral' OR model_name IS NULL")
                connection.execute(sql)
                print("Обновлены значения model_name с 'mistral' на 'mistral:7b-instruct' для существующих записей")
            return
            
        # Добавляем новую колонку model_name
        with engine.begin() as connection:
            sql = text("ALTER TABLE model_configs ADD COLUMN model_name VARCHAR DEFAULT 'mistral:7b-instruct'")
            connection.execute(sql)
            print("Колонка model_name успешно добавлена")
        
        print("Миграция успешно завершена!")
    
    except Exception as e:
        print(f"Ошибка при выполнении миграции: {str(e)}")

if __name__ == "__main__":
    print(f"Начинаем миграцию для добавления поля model_name: {datetime.now()}")
    run_migration()
    print(f"Миграция завершена: {datetime.now()}") 