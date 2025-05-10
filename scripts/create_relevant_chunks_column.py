from sqlalchemy import create_engine, inspect, text
from app.database import Base, SQLALCHEMY_DATABASE_URL

# Универсальный скрипт для добавления колонки relevant_chunks

def add_relevant_chunks_column():
    """Добавляет колонку relevant_chunks в таблицу chats, если она не существует."""
    print("Проверка и добавление колонки relevant_chunks в таблицу chats...")
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    inspector = inspect(engine)
    columns = [column['name'] for column in inspector.get_columns('chats')]
    if 'relevant_chunks' not in columns:
        print("Колонка relevant_chunks не найдена. Добавляем...")
        with engine.connect() as connection:
            connection.execute(text("ALTER TABLE chats ADD COLUMN IF NOT EXISTS relevant_chunks JSON DEFAULT '[]'"))
            connection.commit()
        print("Колонка relevant_chunks успешно добавлена!")
    else:
        print("Колонка relevant_chunks уже существует в таблице chats.")

if __name__ == "__main__":
    add_relevant_chunks_column() 