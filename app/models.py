from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    """Модель пользователя в системе."""
    
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    role = Column(String, default="user")  # Роль: 'user' или 'admin'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Отношения
    chats = relationship("Chat", back_populates="user")
    documents = relationship("Document", back_populates="user")
    model_configs = relationship("ModelConfig", backref="user")


class Chat(Base):
    """Модель для хранения истории чатов пользователя."""
    
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    relevant_chunks = Column(JSON, default=list)  # Для хранения информации о релевантных чанках
    rag_used = Column(Boolean, default=True)  # True = RAG, False = прямой LLM
    
    # Отношение
    user = relationship("User", back_populates="chats")


class Document(Base):
    """Модель для хранения документов, загруженных пользователем."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    source = Column(String, index=True)  # Источник документа (загружен пользователем, импортирован из url, и т.д.)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    # Новые поля для подробной информации
    status = Column(String, default="uploaded")  # uploaded, indexing, indexed, error
    file_name = Column(String)
    file_size = Column(Integer)
    chunks_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    uuid = Column(String, unique=True, index=True)
    chunking_mode = Column(String, default="character")  # Тип чанкинга: character, token, semantic, hierarchical
    
    # Дополнительные метаданные для хранения информации о параметрах обработки
    chunk_size = Column(Integer, nullable=True)  # Размер чанка
    chunk_overlap = Column(Integer, nullable=True)  # Перекрытие между чанками
    embedding_model = Column(String, nullable=True)  # Модель для создания эмбеддингов
    processing_params = Column(JSON, nullable=True)  # Дополнительные параметры обработки в формате JSON
    
    # Отношение
    user = relationship("User", back_populates="documents")


class ModelConfig(Base):
    """Модель для хранения пользовательских настроек генерации."""
    
    __tablename__ = "model_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    temperature = Column(Float, default=0.0)
    top_p = Column(Float, default=0.9)
    max_tokens = Column(Integer, default=2048)
    top_k_chunks = Column(Integer, default=5)
    context_window = Column(Integer, default=8192)
    model_name = Column(String, default="mistral")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 