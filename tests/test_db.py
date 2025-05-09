import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import User, Chat

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5432/postgres"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_create_tables():
    """Тест на создание таблиц в базе данных."""
    # Создаем все таблицы
    Base.metadata.create_all(bind=engine)
    # Проверяем наличие таблиц
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))
        table_names = [row[0] for row in result]
        assert "users" in table_names
        assert "chats" in table_names
        # Создаем пользователя
        conn.execute(
            text("INSERT INTO users (username, hashed_password, disabled) VALUES (:username, :hashed_password, :disabled)"),
            {"username": "testuser", "hashed_password": "testpass", "disabled": False}
        )
        conn.commit()
        user = conn.execute(text("SELECT * FROM users WHERE username = 'testuser'"))
        user = user.first()
        assert user is not None
        assert user[1] == "testuser"
    # Дропаем таблицы после теста
    Base.metadata.drop_all(bind=engine) 