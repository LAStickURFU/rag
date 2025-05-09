from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    question: str
    response: str


class ChatCreate(ChatMessage):
    pass


class Chat(ChatMessage):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    username: str = Field(..., min_length=1)


class UserCreate(UserBase):
    password: str = Field(..., min_length=1)


class User(UserBase):
    id: int
    disabled: bool
    created_at: datetime
    chats: List[Chat] = []

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str


class ModelSettings(BaseModel):
    """Модель для настроек параметров генерации ответов."""
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 1024
    top_k_chunks: int = 5
    context_window: int = 8192  # Максимальный размер контекста (в токенах)


class ChunkInfo(BaseModel):
    """Информация о релевантном фрагменте документа."""
    text: str
    relevance: float
    doc_id: str
    chunk_id: int
    metadata: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    """Расширенный ответ чата с информацией о релевантных чанках."""
    id: int
    user_id: int
    question: str
    response: str
    created_at: datetime
    relevant_chunks: List[ChunkInfo] = []
    meta: Optional[Dict[str, Any]] = None
    rag_used: Optional[bool] = True


# Pydantic модели для оценки RAGAS
class RagasEvalItem(BaseModel):
    question: str
    answer: str  # Ответ, который нужно оценить
    ground_truth: str  # Эталонный ответ


class RagasEvaluationRequest(BaseModel):
    eval_items: List[RagasEvalItem]
    description: Optional[str] = None
    # Можно добавить параметры для RAGAS

