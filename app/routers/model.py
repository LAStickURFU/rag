"""
Маршруты для управления настройками моделей.
"""

import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models import User as UserModel, ModelConfig as ModelConfigModel
from app.schemas import ModelSettings, ModelInfo, AvailableModelsResponse
from app.routers.auth import get_current_user
from app.ollama_client import get_ollama_instance, OllamaLLM

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(prefix="/model", tags=["model"])

# Список доступных моделей и их описания
AVAILABLE_MODELS = [
    {"name": "mistral:7b-instruct", "description": "Mistral 7B Instruct - базовая модель для RAG с широкими возможностями"},
    {"name": "qwen3:8b", "description": "Qwen3 8B - быстрая модель с хорошей производительностью на русском языке"},
    {"name": "llama3:8b", "description": "Llama3 8B - модель от Meta с хорошей генерацией на разных языках"}
]

def get_user_model_config(user_id: int, db: Session) -> ModelConfigModel:
    """Возвращает конфигурацию модели для пользователя или создает её по умолчанию"""
    config = db.query(ModelConfigModel).filter(ModelConfigModel.user_id == user_id).first()
    
    if not config:
        # Создаем конфигурацию по умолчанию
        config = ModelConfigModel(
            user_id=user_id,
            model_name="mistral:7b-instruct"  # Устанавливаем модель по умолчанию
        )
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
        context_window=config.context_window,
        model_name=config.model_name
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
    
    # Проверяем, изменилась ли модель
    model_changed = config.model_name != settings.model_name
    
    # Обновляем настройки
    config.temperature = settings.temperature
    config.top_p = settings.top_p
    config.max_tokens = settings.max_tokens
    config.top_k_chunks = settings.top_k_chunks
    config.context_window = settings.context_window
    config.model_name = settings.model_name
    
    db.commit()
    
    if model_changed:
        logger.info(f"Changed model from {config.model_name} to {settings.model_name} for user {current_user.username}")
        # Сбрасываем кешированный экземпляр Ollama
        _ = get_ollama_instance(settings.model_name, force_new=True)
    
    logger.info(f"Updated model settings for user {current_user.username}")
    
    return settings

@router.get("/available", response_model=AvailableModelsResponse)
async def get_available_models(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Получает список доступных моделей и активную модель.
    """
    # Получаем текущие настройки модели пользователя
    config = get_user_model_config(current_user.id, db)
    
    # Получаем экземпляр Ollama
    ollama = get_ollama_instance(config.model_name)
    
    try:
        # Получаем список доступных моделей с сервера Ollama
        ollama_models = await ollama.async_client.list()
        installed_models = []
        
        if isinstance(ollama_models, dict) and 'models' in ollama_models:
            for model in ollama_models['models']:
                if not isinstance(model, dict):
                    continue
                
                model_name = model.get('name', '')
                installed_models.append(model_name)
                
        # Создаем список моделей с информацией о них
        models = []
        for model_info in AVAILABLE_MODELS:
            name = model_info["name"]
            model_data = ModelInfo(
                name=name,
                description=model_info["description"],
                is_active=(name == config.model_name)
            )
            models.append(model_data)
        
        return AvailableModelsResponse(
            models=models,
            active_model=config.model_name
        )
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        # Возвращаем хотя бы предопределенный список моделей
        models = [
            ModelInfo(
                name=model_info["name"],
                description=model_info["description"],
                is_active=(model_info["name"] == config.model_name)
            )
            for model_info in AVAILABLE_MODELS
        ]
        
        return AvailableModelsResponse(
            models=models,
            active_model=config.model_name
        )

@router.post("/switch/{model_name}", response_model=ModelSettings)
async def switch_model(
    model_name: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Переключает активную модель на указанную.
    """
    # Проверяем, что модель есть в списке доступных
    available_model_names = [model["name"] for model in AVAILABLE_MODELS]
    if model_name not in available_model_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Модель {model_name} не поддерживается. Доступные модели: {', '.join(available_model_names)}"
        )
    
    # Получаем текущие настройки модели пользователя
    config = get_user_model_config(current_user.id, db)
    old_model = config.model_name
    
    # Обновляем модель в настройках
    config.model_name = model_name
    db.commit()
    
    logger.info(f"Switched model from {old_model} to {model_name} for user {current_user.username}")
    
    # Создаем новый экземпляр Ollama с новой моделью
    _ = get_ollama_instance(model_name, force_new=True)
    
    # Возвращаем обновленные настройки
    return ModelSettings(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        top_k_chunks=config.top_k_chunks,
        context_window=config.context_window,
        model_name=config.model_name
    ) 