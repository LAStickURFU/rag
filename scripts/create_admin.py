#!/usr/bin/env python
import os
import sys
import logging
from datetime import datetime

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from sqlalchemy import text
from app.database import engine
from app.routers.auth import get_password_hash

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

def create_admin_user(username="admin", password="adminadmin"):
    """Создает пользователя-администратора с указанными логином и паролем."""
    try:
        with engine.connect() as conn:
            # Начинаем транзакцию
            trans = conn.begin()
            try:
                # Проверяем существование пользователя
                result = conn.execute(text(
                    "SELECT id, role FROM users WHERE username = :username"
                ), {"username": username})
                user = result.fetchone()
                
                hashed_password = get_password_hash(password)
                created_at = datetime.utcnow()
                
                if user:
                    # Пользователь существует, обновляем его роль
                    conn.execute(text(
                        "UPDATE users SET role = 'admin' WHERE username = :username"
                    ), {"username": username})
                    logger.info(f"Updated user '{username}' role to 'admin'")
                else:
                    # Пользователя нет, создаем нового
                    conn.execute(text(
                        "INSERT INTO users (username, hashed_password, disabled, role, created_at) "
                        "VALUES (:username, :password, :disabled, :role, :created_at)"
                    ), {
                        "username": username,
                        "password": hashed_password,
                        "disabled": False,
                        "role": "admin",
                        "created_at": created_at
                    })
                    logger.info(f"Created admin user (username: {username})")
                
                # Фиксируем транзакцию
                trans.commit()
                return True
            except Exception as e:
                # Откатываем транзакцию при ошибке
                trans.rollback()
                logger.error(f"Error in admin user creation transaction: {str(e)}")
                return False
    except Exception as e:
        logger.error(f"Error creating admin user: {str(e)}")
        return False

def ensure_role_column():
    """Проверяет и добавляет колонку role в таблицу users, если она отсутствует."""
    try:
        with engine.connect() as conn:
            # Проверяем наличие колонки 'role' в таблице users
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='users' AND column_name='role'"
            ))
            
            if result.rowcount == 0:
                # Колонки role нет - добавляем её
                conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'"))
                conn.commit()
                logger.info("Added column 'role' to users table")
            else:
                logger.info("Column 'role' already exists in users table")
                
    except Exception as e:
        logger.error(f"Error ensuring role column: {str(e)}")

if __name__ == "__main__":
    # Проверка аргументов командной строки
    username = "admin"
    password = "adminadmin"
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
    if len(sys.argv) > 2:
        password = sys.argv[2]
    
    print(f"Creating admin user: {username}")
    ensure_role_column()
    success = create_admin_user(username, password)
    
    if success:
        print(f"Admin user '{username}' created/updated successfully!")
    else:
        print("Failed to create/update admin user. Check logs for details.")
        sys.exit(1) 