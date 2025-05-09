#!/usr/bin/env python3
"""
Тестирование скрипта оценки RAG.
"""
import json
import asyncio
import argparse
from pathlib import Path

async def test_rag_evaluation(use_builtin_dataset: bool = False, dataset_name: str = None):
    """
    Тестовая функция для проверки работоспособности скрипта оценки RAG.
    
    Args:
        use_builtin_dataset: Использовать ли встроенный датасет
        dataset_name: Название встроенного датасета (sberquad, RuBQ)
    """
    # Импортируем модуль оценки
    from scripts.evaluate_rag import RAGEvaluator
    
    try:
        # Создаем экземпляр оценщика
        evaluator = RAGEvaluator(
            ollama_model_name="mistral:7b-instruct",
            embedding_model_name="intfloat/multilingual-e5-base"
        )
        
        print("RAG evaluator initialized successfully")
        print(f"Available built-in datasets: {list(evaluator.builtin_datasets.keys())}")
        
        if use_builtin_dataset:
            if dataset_name and dataset_name in evaluator.builtin_datasets:
                # Используем указанный встроенный датасет
                print(f"Using built-in dataset: {dataset_name}")
                eval_data = await evaluator.load_evaluation_data(dataset_name)
            elif evaluator.builtin_datasets:
                # Используем первый доступный встроенный датасет
                dataset_name = next(iter(evaluator.builtin_datasets.keys()))
                print(f"No dataset name specified. Using built-in dataset: {dataset_name}")
                eval_data = await evaluator.load_evaluation_data(dataset_name)
            else:
                # Если нет встроенных датасетов, используем тестовые данные
                print("No built-in datasets available. Using test data.")
                use_builtin_dataset = False
        
        if not use_builtin_dataset:
            # Создаем тестовые данные
            test_data = [
                {
                    "question": "Что такое RAG?",
                    "answer": "RAG (Retrieval Augmented Generation) — это технология, объединяющая поиск информации и генерацию текста. Система ищет релевантные данные в базе знаний и использует их для улучшения ответов языковой модели.",
                    "ground_truth": "RAG (Retrieval Augmented Generation) — это подход, при котором генеративная языковая модель дополняется системой поиска. Сначала находятся релевантные документы, затем на их основе модель генерирует точный и информативный ответ."
                },
                {
                    "question": "Какие компоненты входят в улучшенную RAG-систему?",
                    "answer": "В улучшенную RAG-систему входят: умный чанкер для разбиения документов, гибридный поиск (векторный поиск + BM25), переранжирование с помощью CrossEncoder и адаптивный выбор количества документов для контекста.",
                    "ground_truth": "Улучшенная RAG-система включает гибридный поиск (Dense + Sparse Retrieval), переранжирование с CrossEncoder, улучшенный чанкинг с механизмами отказоустойчивости и адаптивный выбор количества документов для контекста."
                }
            ]
            
            # Сохраняем тестовые данные во временный файл
            temp_data_path = Path("temp_test_data.json")
            with open(temp_data_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            print(f"Created test data with {len(test_data)} items")
            
            # Загружаем тестовые данные
            eval_data = await evaluator.load_evaluation_data(str(temp_data_path))
        
        # Запускаем оценку с тестовым набором
        print("Running evaluation...")
        results = await evaluator.run_evaluation(eval_data)
        
        # Выводим результаты
        print("\nTest evaluation results:")
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        
        # Визуализируем и сохраняем результаты
        dataset_name_for_save = dataset_name if use_builtin_dataset else None
        plot_path = evaluator.visualize_metrics(results)
        json_path = evaluator.save_results(results, eval_data, dataset_name_for_save)
        
        print(f"\nResults saved to {json_path}")
        print(f"Plot saved to {plot_path}")
        
        print("\nTest completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Удаляем временный файл, если он был создан
        if not use_builtin_dataset:
            temp_data_path = Path("temp_test_data.json")
            if temp_data_path.exists():
                temp_data_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG evaluation script")
    parser.add_argument("--builtin", action="store_true", help="Use built-in dataset")
    parser.add_argument("--dataset", type=str, choices=["sberquad", "RuBQ"], 
                        help="Name of built-in dataset to use")
    args = parser.parse_args()
    
    asyncio.run(test_rag_evaluation(use_builtin_dataset=args.builtin, dataset_name=args.dataset)) 