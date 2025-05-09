"""
Модуль с декораторами для отслеживания производительности функций в RAG-системе.
"""

import time
import functools
import inspect
import logging
import asyncio
from typing import Callable, Any, TypeVar, cast

from app.monitoring.metrics import get_metrics_instance

# Настройка логирования
logger = logging.getLogger(__name__)

# Типовые переменные для более точных аннотаций
F = TypeVar("F", bound=Callable[..., Any])
AsyncF = TypeVar("AsyncF", bound=Callable[..., Any])

def track_latency(phase: str) -> Callable[[F], F]:
    """
    Декоратор для отслеживания времени выполнения функций и записи метрик.
    
    Args:
        phase: Фаза RAG для метрики (retrieval, generation, total)
        
    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        # Проверяем, является ли функция асинхронной
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_instance()
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Записываем метрику
                    metrics.record_latency(phase, elapsed_ms)
                    
                    logger.debug(f"{func.__name__} ({phase}) took {elapsed_ms:.2f}ms")
                    return result
                    
                except Exception as e:
                    # Записываем ошибку
                    metrics.record_error(f"{phase}_errors")
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    raise e
                    
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_instance()
                
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Записываем метрику
                    metrics.record_latency(phase, elapsed_ms)
                    
                    logger.debug(f"{func.__name__} ({phase}) took {elapsed_ms:.2f}ms")
                    return result
                    
                except Exception as e:
                    # Записываем ошибку
                    metrics.record_error(f"{phase}_errors")
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    raise e
                    
            return cast(F, sync_wrapper)
            
    return decorator

def track_query(func: F) -> F:
    """
    Декоратор для отслеживания пользовательских запросов.
    
    Args:
        func: Функция для декорирования
        
    Returns:
        Декорированная функция
    """
    is_async = asyncio.iscoroutinefunction(func)
    
    if is_async:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics_instance()
            
            # Ищем параметр запроса - обычно это первый аргумент или query/question в kwargs
            query = None
            if len(args) > 1:
                query = args[1]  # Предполагаем, что первый arg - self, второй - query
            elif "query" in kwargs:
                query = kwargs["query"]
            elif "question" in kwargs:
                query = kwargs["question"]
                
            if query and isinstance(query, str):
                # Записываем запрос
                metrics.record_query(query)
                
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                metrics.record_error("total_errors")
                raise e
                
        return cast(F, async_wrapper)
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics_instance()
            
            # Ищем параметр запроса
            query = None
            if len(args) > 1:
                query = args[1]  # Предполагаем, что первый arg - self, второй - query
            elif "query" in kwargs:
                query = kwargs["query"]
            elif "question" in kwargs:
                query = kwargs["question"]
                
            if query and isinstance(query, str):
                # Записываем запрос
                metrics.record_query(query)
                
            try:
                return func(*args, **kwargs)
            except Exception as e:
                metrics.record_error("total_errors")
                raise e
                
        return cast(F, sync_wrapper)

def track_quality(metric_name: str) -> Callable[[F], F]:
    """
    Декоратор для отслеживания метрик качества.
    
    Args:
        metric_name: Название метрики качества
        
    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_instance()
                
                result = await func(*args, **kwargs)
                
                # Предполагаем, что результат - число или словарь с нужной метрикой
                if isinstance(result, (int, float)):
                    metrics.record_quality(metric_name, float(result))
                elif isinstance(result, dict) and metric_name in result:
                    metrics.record_quality(metric_name, float(result[metric_name]))
                
                return result
                
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                metrics = get_metrics_instance()
                
                result = func(*args, **kwargs)
                
                # Предполагаем, что результат - число или словарь с нужной метрикой
                if isinstance(result, (int, float)):
                    metrics.record_quality(metric_name, float(result))
                elif isinstance(result, dict) and metric_name in result:
                    metrics.record_quality(metric_name, float(result[metric_name]))
                
                return result
                
            return cast(F, sync_wrapper)
            
    return decorator 