#!/usr/bin/env python3
"""
Скрипт для запуска оценки качества RAG.

Использование:
    python -m scripts.run_evaluation --data sberquad --model mistral:7b-instruct
    python -m scripts.run_evaluation --list-datasets
    python -m scripts.run_evaluation --data custom.json --model mistral
    python -m scripts.run_evaluation --retrieval-only --data sberquad
    python -m scripts.run_evaluation --generation-only --data sberquad
"""
import sys
import os
import asyncio
import argparse
import random  # Добавляем импорт random для выбора случайных элементов
from pathlib import Path
import logging

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.logging_config import setup_logging
from scripts.evaluate_rag import RAGEvaluator, get_builtin_datasets

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--data", type=str, help="Dataset name (sberquad, RuBQ) or path to JSON file")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct", help="Ollama model name")
    parser.add_argument("--embeddings", type=str, default="intfloat/multilingual-e5-base", 
                        help="Embedding model name")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sample selection (for reproducibility)")
    parser.add_argument("--retrieval-only", action="store_true", 
                        help="Evaluate only retrieval performance")
    parser.add_argument("--generation-only", action="store_true", 
                        help="Evaluate only generation performance")
    parser.add_argument("--isolated-index", action="store_true", 
                        help="Create isolated index for evaluation")
    args = parser.parse_args()

    # Создаем экземпляр оценщика
    evaluator = RAGEvaluator(
        ollama_model_name=args.model,
        embedding_model_name=args.embeddings
    )
    
    # Если запрошен список датасетов, выводим его и завершаем работу
    if args.list_datasets:
        print("\nДоступные датасеты:")
        for name, path in evaluator.builtin_datasets.items():
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    count = len(json.load(f))
                print(f"  - {name}: {path} ({count} примеров)")
            except Exception as e:
                print(f"  - {name}: {path} (ошибка чтения: {e})")
        return 0
    
    # Проверяем, указан ли датасет
    if not args.data:
        if evaluator.builtin_datasets:
            dataset_name = next(iter(evaluator.builtin_datasets.keys()))
            print(f"\nДатасет не указан. Используем встроенный датасет: {dataset_name}")
            args.data = dataset_name
        else:
            print("\nОшибка: датасет не указан, а встроенные датасеты недоступны.")
            print("Используйте параметр --data для указания пути к файлу или названия встроенного датасета.")
            return 1
    
    # Загружаем данные
    try:
        print(f"Загрузка данных из {args.data}...")
        eval_data = await evaluator.load_evaluation_data(args.data)
        
        # Устанавливаем seed для воспроизводимости, если указан
        if args.seed is not None:
            random.seed(args.seed)
            print(f"Установлен seed для случайной выборки: {args.seed}")
        
        # Ограничиваем количество примеров, если указано
        if args.limit and args.limit > 0 and args.limit < len(eval_data):
            # Выбираем случайные элементы вместо первых n
            eval_data = random.sample(eval_data, args.limit)
            print(f"Ограничили набор до {args.limit} случайных примеров")
        
        print(f"Загружено {len(eval_data)} примеров для оценки")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return 1
    
    # Запускаем оценку
    try:
        dataset_name = args.data if args.data in evaluator.builtin_datasets else None
        
        if args.retrieval_only:
            # Оценка только компонента поиска
            print(f"Запуск оценки поиска (retrieval) с использованием модели {args.embeddings}...")
            questions = [item['question'] for item in eval_data]
            ground_truths = [item.get('ground_truth', '') for item in eval_data]
            
            # Создаем изолированный индекс, если указан параметр
            contexts = None
            if args.isolated_index and all('context' in item for item in eval_data):
                contexts = [item['context'] for item in eval_data if 'context' in item]
                
            results = await evaluator.evaluate_retrieval(
                questions=questions, 
                ground_truths=ground_truths,
                contexts=contexts,
                top_k=3
            )
            
            print("\nРезультаты оценки поиска (retrieval):")
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
                
            # Сохраняем результаты
            plot_path = evaluator.visualize_metrics(results)
            json_path = evaluator.save_results(results, eval_data, f"{dataset_name}_retrieval_only")
            
        elif args.generation_only:
            # Оценка только компонента генерации
            print(f"Запуск оценки генерации (generation) с использованием модели {args.model}...")
            questions = [item['question'] for item in eval_data]
            ground_truths = [item.get('ground_truth', '') for item in eval_data]
            
            # Получаем контексты через поиск
            contexts = []
            print("Получение контекстов для оценки генерации...")
            for i, question in enumerate(questions):
                retrieved_chunks_with_scores = evaluator.rag_service.search(question, top_k=3)
                retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
                contexts.append(retrieved_contexts)
                if i % 5 == 0:
                    print(f"  Обработано {i+1}/{len(questions)} вопросов")
            
            # Если в данных есть ответы, используем их
            answers = [item.get('answer', '') for item in eval_data]
            if not any(answers):
                answers = None
            
            results = await evaluator.evaluate_generation(
                questions=questions,
                contexts=contexts,
                ground_truths=ground_truths,
                answers=answers
            )
            
            print("\nРезультаты оценки генерации (generation):")
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
                
            # Сохраняем результаты
            plot_path = evaluator.visualize_metrics(results)
            json_path = evaluator.save_results(results, eval_data, f"{dataset_name}_generation_only")
            
        else:
            # Полная оценка
            print(f"Запуск полной оценки RAG с использованием модели {args.model}...")
            if args.isolated_index:
                print("Используется изолированный индекс для оценки")
            results = await evaluator.run_evaluation(
                eval_data,
                use_isolated_index=args.isolated_index
            )
            
            print("\nРезультаты полной оценки RAG:")
            for metric, score in results.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric}: {score:.4f}")
                else:
                    print(f"  {metric}: {score}")
            
            # Добавляем вывод примеров с ответами системы
            if '_detailed_samples' in results:
                print("\n=-=-=-=-= Примеры оценки RAG =-=-=-=-=")
                for i, example in enumerate(results['_detailed_samples']):
                    print(f"\n[Пример {i+1}/{len(results['_detailed_samples'])}]")
                    print(f"Вопрос: {example['question']}")
                    print(f"Эталонный ответ: {example['ground_truth']}")
                    print(f"Ответ RAG-системы: {example['response']}")
                    print(f"Время поиска: {example.get('search_time_sec', 0):.2f} сек.")
                    print(f"Время генерации: {example.get('generation_time_sec', 0):.2f} сек.")
                    print(f"Найдено {len(example.get('retrieved_contexts', []))} контекстов")
                    print("-" * 70)
            
            # Сохраняем результаты
            plot_path = evaluator.visualize_metrics(results)
            json_path = evaluator.save_results(results, eval_data, dataset_name)
        
        print(f"\nГрафик сохранен: {plot_path}")
        print(f"Результаты сохранены: {json_path}")
        
        return 0
    except Exception as e:
        import traceback
        print(f"Ошибка при выполнении оценки: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 