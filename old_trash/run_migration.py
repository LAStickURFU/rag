#!/usr/bin/env python3
"""
Скрипт для выполнения SQL-миграции базы данных.
Запускается вручную при необходимости внесения изменений в схему базы данных.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
load_dotenv()

def run_migration(sql_file_path):
    """
    Выполняет SQL-скрипт миграции.
    
    Args:
        sql_file_path: Путь к SQL-файлу для выполнения
    """
    if not os.path.exists(sql_file_path):
        logger.error(f"Файл {sql_file_path} не найден")
        return False
    
    # Получаем строку подключения к базе данных
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL не определен в переменных окружения")
        return False
    
    logger.info(f"Выполняем миграцию из файла {sql_file_path}")
    
    try:
        # Создаем подключение к базе данных
        engine = create_engine(database_url)
        
        # Читаем содержимое SQL-файла
        with open(sql_file_path, "r") as file:
            sql_script = file.read()
        
        # Выполняем SQL-скрипт
        with engine.connect() as connection:
            connection.execute(text(sql_script))
            connection.commit()
        
        logger.info("Миграция успешно выполнена")
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при выполнении миграции: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Использование: python {sys.argv[0]} <path_to_sql_file>")
        sys.exit(1)
    
    sql_file = sys.argv[1]
    success = run_migration(sql_file)
    
    if not success:
        sys.exit(1) 