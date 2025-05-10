"""
Маршруты для управления настройками моделей.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User as UserModel, ModelConfig as ModelConfigModel
from app.schemas import ModelSettings
from app.routers.auth import get_current_user

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(prefix="/model", tags=["model"])

def get_user_model_config(user_id: int, db: Session) -> ModelConfigModel:
    """Возвращает конфигурацию модели для пользователя или создает её по умолчанию"""
    config = db.query(ModelConfigModel).filter(ModelConfigModel.user_id == user_id).first()
    
    if not config:
        # Создаем конфигурацию по умолчанию
        config = ModelConfigModel(user_id=user_id)
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return config

@router.get("/settings", response_model=ModelSettings)
async def get_model_settings(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получает текущие настройки модели для пользователя.
    """
    config = get_user_model_config(current_user.id, db)
    
    return ModelSettings(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        top_k_chunks=config.top_k_chunks,
        context_window=config.context_window
    )

@router.post("/settings", response_model=ModelSettings)
async def update_model_settings(
    settings: ModelSettings,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Обновляет настройки модели для пользователя.
    """
    config = get_user_model_config(current_user.id, db)
    
    # Обновляем настройки
    config.temperature = settings.temperature
    config.top_p = settings.top_p
    config.max_tokens = settings.max_tokens
    config.top_k_chunks = settings.top_k_chunks
    config.context_window = settings.context_window
    
    db.commit()
    
    logger.info(f"Updated model settings for user {current_user.username}")
    
    return settings 