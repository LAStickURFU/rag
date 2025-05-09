#!/usr/bin/env python
"""
Скрипт для тестирования работы RAG-системы с загруженными документами.
Позволяет задавать вопросы и получать ответы на основе индексированных документов.
"""

import asyncio
import logging
import argparse
from app.rag import RAGService
from app.ollama_client import get_ollama_instance

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ask_rag(query: str, top_k: int = 5, verbose: bool = False):
    """
    Выполняет запрос к RAG-системе и возвращает ответ
    
    Args:
        query: Запрос пользователя
        top_k: Количество релевантных фрагментов для генерации ответа
        verbose: Флаг для вывода подробной информации
    
    Returns:
        Ответ от RAG-системы
    """
    # Инициализация RAG-сервиса
    rag_service = RAGService()
    
    # Поиск релевантных фрагментов
    search_results = rag_service.search(query, top_k=top_k)
    
    if not search_results:
        return "Не найдено релевантных фрагментов для этого запроса."
    
    # Вывод найденных фрагментов, если включен подробный режим
    if verbose:
        logger.info(f"Найдено {len(search_results)} релевантных фрагментов:")
        for i, (chunk, score) in enumerate(search_results):
            logger.info(f"Фрагмент {i+1} (релевантность: {score:.4f}):")
            logger.info(f"  Документ: {chunk.metadata.get('title', 'Неизвестный')}")
            logger.info(f"  Текст: {chunk.text[:150]}...")
            logger.info("---")
    
    # Генерация промпта
    prompt = rag_service.generate_prompt(query, top_k_chunks=top_k)
    
    # Получение ответа от LLM
    ollama = get_ollama_instance()
    try:
        await ollama.ensure_model_loaded()
        response = await ollama.generate(prompt)
        return response
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        return f"Произошла ошибка при генерации ответа: {str(e)}"

async def interactive_mode():
    """Запускает интерактивный режим для тестирования RAG-системы"""
    print("\n=== RAG-система: Интерактивный режим ===")
    print("Введите вопрос для получения ответа или 'выход' для завершения\n")
    
    while True:
        query = input("\nВопрос: ").strip()
        
        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            print("Завершение работы...")
            break
        
        if not query:
            continue
        
        print("\nПоиск ответа...")
        
        try:
            response = await ask_rag(query, verbose=True)
            print("\n=== Ответ ===")
            print(response)
            print("=============")
        except Exception as e:
            print(f"Ошибка: {str(e)}")

async def single_query_mode(query: str, verbose: bool = False):
    """Выполняет одиночный запрос к RAG-системе"""
    print(f"\n=== RAG-система: Запрос '{query}' ===\n")
    
    try:
        response = await ask_rag(query, verbose=verbose)
        print("\n=== Ответ ===")
        print(response)
        print("=============")
    except Exception as e:
        print(f"Ошибка: {str(e)}")

async def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Тестирование RAG-системы")
    parser.add_argument('-q', '--query', type=str, help="Одиночный запрос к системе")
    parser.add_argument('-v', '--verbose', action='store_true', help="Вывод подробной информации")
    
    args = parser.parse_args()
    
    if args.query:
        await single_query_mode(args.query, verbose=args.verbose)
    else:
        await interactive_mode()

if __name__ == "__main__":
    asyncio.run(main()) 