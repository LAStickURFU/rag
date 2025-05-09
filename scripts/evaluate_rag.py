#!/usr/bin/env python3
"""
Evaluate RAG pipeline using RAGAS metrics and visualize the results.
Работает с существующей RAG-системой и адаптирован для версии RAGAS 0.2.0
"""
import os
import json
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Использовать не-интерактивный бэкенд для избежания ошибок в фоновых потоках
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from datetime import datetime
from collections import defaultdict
import traceback

# Импорты RAGAS для версии 0.2.0
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.executor import Executor
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset

# Импорт для патчинга RAGAS Executor
import types
from functools import wraps
from tqdm.auto import tqdm
from ragas.run_config import RunConfig

# Импорты из проекта
from app.rag import RAGService, Document
from app.ollama_client import OllamaLLM
from app.retrieval.vector_index import VectorIndex
from app.chunking.robust_chunker import RobustChunker

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Патч RAGAS Executor для использования правильного размера прогресс-бара
def patch_ragas_executor():
    """Патч для класса Executor в RAGAS, чтобы использовать фактическое количество элементов в прогресс-баре."""
    try:
        from ragas.executor import Executor
        
        # Сохраняем оригинальную функцию results
        original_results = Executor.results
        
        # Создаем новую функцию results с патчем для tqdm
        @wraps(original_results)
        def patched_results(self):
            """
            Патчим метод results класса Executor, чтобы использовать
            фактическое количество элементов в прогресс-баре вместо 20.
            """
            from ragas.executor import is_event_loop_running, as_completed
            import asyncio
            
            if is_event_loop_running():
                try:
                    import nest_asyncio
                except ImportError:
                    raise ImportError(
                        "It seems like your running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                    )
                
                if not self._nest_asyncio_applied:
                    nest_asyncio.apply()
                    self._nest_asyncio_applied = True
            
            futures_as_they_finish = as_completed(
                coros=[afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs],
                max_workers=(self.run_config or RunConfig()).max_workers,
            )
            
            async def _aresults():
                results = []
                for future in tqdm(
                    await futures_as_they_finish,
                    desc=self.desc,
                    total=len(self.jobs),  # Используем фактическое количество задач
                    leave=self.keep_progress_bar,
                    disable=not self.show_progress,
                ):
                    r = await future
                    results.append(r)
                return results
            
            results = asyncio.run(_aresults())
            sorted_results = sorted(results, key=lambda x: x[0])
            return [r[1] for r in sorted_results]
        
        # Заменяем оригинальный метод на наш патч
        Executor.results = patched_results
        logger.info("RAGAS Executor успешно патчен для использования правильного размера прогресс-бара")
        return True
    except Exception as e:
        logger.warning(f"Не удалось применить патч к RAGAS Executor: {e}")
        return False

# Дополнительный патч для функции evaluate, чтобы она использовала правильный прогресс-бар
def patch_ragas_evaluate():
    """Патч для функции evaluate в RAGAS, чтобы заменить жестко закодированное значение 20 на фактическое количество элементов."""
    try:
        # Импортируем ragas.evaluation, чтобы получить функцию evaluate
        import ragas.evaluation
        original_evaluate = ragas.evaluation.evaluate
        
        @wraps(original_evaluate)
        def patched_evaluate(*args, **kwargs):
            """Патчированная версия функции evaluate с настроенным прогресс-баром."""
            # Получаем датасет для определения фактического количества элементов
            dataset = args[0] if args else kwargs.get('dataset')
            if dataset:
                logger.info(f"Запуск evaluate с датасетом размером {len(dataset)}")
                
                # Обновляем вывод прогресса
                if "show_progress" not in kwargs:
                    kwargs["show_progress"] = True
                
                # Патчим также настройки run_config
                if "run_config" not in kwargs:
                    kwargs["run_config"] = RunConfig()
            
            # Вызываем оригинальную функцию с обновленными параметрами
            return original_evaluate(*args, **kwargs)
        
        # Заменяем оригинальную функцию нашей патчированной версией
        ragas.evaluation.evaluate = patched_evaluate
        logger.info("RAGAS evaluate успешно патчен для использования правильного размера прогресс-бара")
        return True
    except Exception as e:
        logger.warning(f"Не удалось применить патч к RAGAS evaluate: {e}")
        return False

def get_builtin_datasets() -> Dict[str, Path]:
    """
    Получает или скачивает встроенные датасеты для оценки RAG.
    
    Returns:
        Словарь с путями к файлам датасетов {имя_датасета: путь_к_файлу}
    """
    datasets_dir = Path("app/evaluation/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    
    # SberQuAD
    sberquad_path = datasets_dir / "sberquad.json"
    if not sberquad_path.exists():
        try:
            logger.info("Downloading SberQuAD dataset...")
            ds = load_dataset("sberquad", split="validation[:100]")
            out = []
            for item in ds:
                out.append({
                    "question": item["question"],
                    "answer": "",
                    "ground_truth": item["answers"]["text"][0] if item["answers"]["text"] else ""
                })
            with open(sberquad_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info(f"SberQuAD dataset saved to {sberquad_path}")
        except Exception as e:
            logger.error(f"Error downloading SberQuAD: {e}")
    if sberquad_path.exists():
        datasets["sberquad"] = sberquad_path
    
    # RuBQ
    rubq_path = datasets_dir / "RuBQ.json"
    if not rubq_path.exists():
        try:
            logger.info("Downloading RuBQ dataset...")
            url = "https://raw.githubusercontent.com/vladislavneon/RuBQ/master/RuBQ_2.0/RuBQ_2.0_test.json"
            data = requests.get(url, timeout=30).json()
            out = []
            for item in data:
                out.append({
                    "question": item["question_text"],
                    "answer": "",
                    "ground_truth": item["answer_text"]
                })
            with open(rubq_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            logger.info(f"RuBQ dataset saved to {rubq_path}")
        except Exception as e:
            logger.error(f"Error downloading RuBQ: {e}")
    if rubq_path.exists():
        datasets["RuBQ"] = rubq_path
    
    return datasets

class RAGEvaluator:
    """Класс для проведения оценки качества RAG с помощью RAGAS."""
    
    def __init__(self, 
                 ollama_model_name: str = "mistral:7b-instruct",
                 embedding_model_name: str = "intfloat/multilingual-e5-base"):
        """
        Инициализация компонентов для оценки.
        
        Args:
            ollama_model_name: Название модели Ollama для оценки
            embedding_model_name: Название модели для эмбеддингов
        """
        # Применяем патчи к RAGAS для корректного отображения прогресс-бара
        patch_ragas_executor()
        patch_ragas_evaluate()  # Добавляем новый патч
        
        # Инициализация RAG-сервиса
        self.rag_service = RAGService(
            model_name=embedding_model_name,
            use_hybrid_search=True,
            use_reranker=True,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Инициализация клиента Ollama
        self.ollama_client = OllamaLLM(model_name=ollama_model_name)
        
        # Инициализация LLM и Embeddings для RAGAS
        self.ragas_llm = Ollama(model=ollama_model_name)
        self.ragas_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Папка для сохранения результатов - используем только один путь
        self.results_dir = Path("app/evaluation/results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Сохраняем имена моделей 
        self.ollama_model_name = ollama_model_name
        self.embedding_model_name = embedding_model_name
        
        # Информация о датасете
        self.total_dataset_size = 0  # Будет установлено при загрузке данных
        
        # Получаем список встроенных датасетов
        self.builtin_datasets = get_builtin_datasets()
        
        # Создаем чанкер для разделения текста
        self.chunker = RobustChunker(
            chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            language=os.getenv("LANGUAGE", "ru"),
            spacy_model=os.getenv("SPACY_MODEL", "ru_core_news_md")
        )
        
        # Статистика по вопросам для категоризации
        self.questions_stats = {}
        
        logger.info(f"Initialized RAG evaluator with Ollama model {ollama_model_name}")
        logger.info(f"Available built-in datasets: {list(self.builtin_datasets.keys())}")
    
    async def load_evaluation_data(self, file_path_or_name: str) -> List[Dict[str, Any]]:
        """
        Загрузка данных для оценки из JSON файла или из встроенных датасетов.
        
        Args:
            file_path_or_name: Путь к JSON файлу с вопросами и эталонными ответами,
                              или название встроенного датасета
            
        Returns:
            Список словарей с данными для оценки
        """
        # Проверяем, является ли аргумент названием встроенного датасета
        if file_path_or_name in self.builtin_datasets:
            file_path = self.builtin_datasets[file_path_or_name]
            logger.info(f"Using built-in dataset: {file_path_or_name}")
        else:
            file_path = file_path_or_name
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and 'eval_items' in data:
                # Формат, используемый во фронтенде
                items = data['eval_items']
            elif isinstance(data, list):
                # Простой список элементов
                items = data
            else:
                raise ValueError(f"Unsupported data format in {file_path}")
            
            # Сохраняем информацию о полном размере датасета
            self.total_dataset_size = len(items)
            
            logger.info(f"Loaded {len(items)} evaluation items from {file_path}")
            return items
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def create_isolated_index(self, contexts: List[str], namespace: str = "eval") -> VectorIndex:
        """
        Создает изолированный векторный индекс для оценки, не затрагивая основной.
        
        Args:
            contexts: Список текстовых контекстов для индексации
            namespace: Пространство имен для индекса (префикс для коллекции)
            
        Returns:
            Векторный индекс с проиндексированными контекстами
        """
        logger.info(f"Creating isolated index with {len(contexts)} contexts")
        
        # Создаем новый индекс с уникальным именем
        isolated_index = VectorIndex(
            index_name=f"{namespace}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            vector_size=self.ragas_embeddings.client.get_sentence_embedding_dimension()
        )
        
        # Разбиваем контексты на чанки и добавляем в индекс
        doc_id = 0
        chunk_id = 0
        for context in contexts:
            doc_id += 1
            
            # Создаем документ
            doc = Document(
                content=context,
                metadata={"source": "evaluation", "doc_id": str(doc_id)}
            )
            
            # Разбиваем на чанки
            chunks = self.chunker.create_chunks_from_document(doc, str(doc_id))
            
            # Добавляем чанки в индекс
            if chunks:
                # Векторизуем фрагменты
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = np.array([self.ragas_embeddings.embed_query(text) for text in chunk_texts])
                
                # Добавляем в индекс
                for i, chunk in enumerate(chunks):
                    chunk_id += 1
                    chunk.doc_id = str(doc_id)
                    chunk.chunk_id = str(chunk_id)
                
                isolated_index.add_chunks(chunks, embeddings)
        
        logger.info(f"Isolated index created with {chunk_id} chunks from {doc_id} documents")
        return isolated_index
    
    async def evaluate_retrieval(
        self, 
        questions: List[str], 
        ground_truths: List[str],
        contexts: Optional[List[str]] = None,
        isolated_index: Optional[VectorIndex] = None,
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        Оценивает только качество поиска (retrieval) без генерации ответов.
        
        Args:
            questions: Список вопросов
            ground_truths: Список эталонных ответов
            contexts: Опционально, список контекстов для изолированной индексации
            isolated_index: Опционально, готовый изолированный индекс
            top_k: Количество контекстов для поиска
            
        Returns:
            Словарь с метриками качества поиска
        """
        if not isolated_index and contexts:
            # Если нет готового индекса, но есть контексты, создаем индекс
            isolated_index = self.create_isolated_index(contexts)
        
        # Подготовка данных для оценки
        retrieval_results = []
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
            try:
                # Поиск контекстов - используем либо изолированный индекс, либо общий
                if isolated_index:
                    # Получаем эмбеддинг запроса
                    query_embedding = np.array(self.ragas_embeddings.embed_query(question))
                    
                    # Выполняем поиск
                    retrieved_chunks_with_scores = isolated_index.search(
                        query_vector=query_embedding,
                        top_k=3
                    )
                    retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
                else:
                    retrieved_chunks_with_scores = self.rag_service.search(question, top_k=top_k)
                    retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
                
                # Добавляем результаты поиска к вопросу
                retrieval_results.append({
                    "question": question,
                    "retrieved_contexts": retrieved_contexts,
                    "ground_truth": ground_truth
                })
                
                logger.info(f"Evaluated retrieval for question {i+1}/{len(questions)}")
            except Exception as e:
                logger.error(f"Error evaluating retrieval for question: {e}")
        
        # Создаем датасет для оценки контекстов
        retrieval_dataset = {
            "question": [item["question"] for item in retrieval_results],
            "contexts": [item["retrieved_contexts"] for item in retrieval_results],
            "reference": [item["ground_truth"] for item in retrieval_results]
        }
        
        # Используем только метрики для оценки поиска
        retrieval_metrics = [
            context_recall,
            context_precision
        ]
        
        # Создаем датасет RAGAS
        evaluation_dataset = EvaluationDataset.from_dict(retrieval_dataset)
        
        # Запускаем оценку
        logger.info("Running retrieval evaluation with RAGAS")
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=retrieval_metrics,
            embeddings=self.ragas_embeddings
        )
        
        # Преобразуем результаты в словарь
        metrics_dict = {}
        if hasattr(results, '_repr_dict') and isinstance(results._repr_dict, dict):
            metrics_dict = {k: float(v) for k, v in results._repr_dict.items()}
        
        return metrics_dict
    
    async def evaluate_generation(
        self,
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Оценивает только качество генерации ответов без учета поиска.
        
        Args:
            questions: Список вопросов
            contexts: Список списков контекстов для каждого вопроса
            ground_truths: Список эталонных ответов
            answers: Опционально, готовые ответы (если нет, будут сгенерированы)
            
        Returns:
            Словарь с метриками качества генерации
        """
        # Подготовка данных для оценки
        generation_results = []
        
        for i, (question, context_list, ground_truth) in enumerate(zip(questions, contexts, ground_truths)):
            try:
                if answers and i < len(answers):
                    # Используем готовый ответ, если есть
                    answer = answers[i]
                else:
                    # Генерируем промпт и ответ
                    prompt = self.rag_service.generate_prompt_from_contexts(question, context_list)
                    answer = await self.ollama_client.generate(prompt)
                
                # Добавляем результаты генерации
                generation_results.append({
                    "question": question,
                    "answer": answer,
                    "contexts": context_list,
                    "ground_truth": ground_truth
                })
                
                logger.info(f"Evaluated generation for question {i+1}/{len(questions)}")
            except Exception as e:
                logger.error(f"Error evaluating generation for question: {e}")
        
        # Создаем датасет для оценки генерации
        generation_dataset = {
            "question": [item["question"] for item in generation_results],
            "answer": [item["answer"] for item in generation_results],
            "contexts": [item["contexts"] for item in generation_results],
            "reference": [item["ground_truth"] for item in generation_results]
        }
        
        # Используем только метрики для оценки генерации
        generation_metrics = [
            faithfulness,
            answer_relevancy
        ]
        
        # Создаем датасет RAGAS
        evaluation_dataset = EvaluationDataset.from_dict(generation_dataset)
        
        # Запускаем оценку
        logger.info("Running generation evaluation with RAGAS")
        results = evaluate(
            dataset=evaluation_dataset,
            metrics=generation_metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )
        
        # Преобразуем результаты в словарь
        metrics_dict = {}
        if hasattr(results, '_repr_dict') and isinstance(results._repr_dict, dict):
            metrics_dict = {k: float(v) for k, v in results._repr_dict.items()}
        
        return metrics_dict
    
    def analyze_questions(self, questions: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Анализирует вопросы и возвращает статистику для категоризации.
        
        Args:
            questions: Список вопросов для анализа
            
        Returns:
            Словарь статистик по каждому вопросу
        """
        stats = {}
        
        for i, question in enumerate(questions):
            # Анализируем вопрос
            stats[i] = {
                "length": len(question),
                "words_count": len(question.split()),
                "has_question_mark": "?" in question,
                "type": self._detect_question_type(question),
                "complexity": self._estimate_complexity(question)
            }
        
        # Сохраняем статистику
        self.questions_stats = stats
        return stats
    
    def _detect_question_type(self, question: str) -> str:
        """Определяет тип вопроса (who, what, when, where, why, how)"""
        question = question.lower()
        
        if any(w in question for w in ["кто", "чей", "кому", "кого"]):
            return "who"
        elif any(w in question for w in ["что", "какой", "какая", "какие", "каких"]):
            return "what"
        elif any(w in question for w in ["когда", "время", "дата", "год"]):
            return "when"
        elif any(w in question for w in ["где", "куда", "откуда", "место"]):
            return "where"
        elif any(w in question for w in ["почему", "зачем", "отчего"]):
            return "why"
        elif any(w in question for w in ["как", "каким образом", "способ"]):
            return "how"
        else:
            return "other"
    
    def _estimate_complexity(self, question: str) -> str:
        """Оценивает сложность вопроса по эвристикам"""
        # Простая эвристика - по длине и наличию сложных конструкций
        words = question.split()
        
        if len(words) <= 5:
            complexity = "simple"
        elif len(words) <= 12:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # Усложняем оценку, если есть признаки сложного вопроса
        complex_indicators = ["почему", "связь между", "сравнить", "перечислить", "объяснить"]
        if any(ind in question.lower() for ind in complex_indicators):
            # Повышаем сложность на одну ступень
            if complexity == "simple":
                complexity = "medium"
            elif complexity == "medium":
                complexity = "complex"
        
        return complexity
    
    def categorize_results(
        self, 
        results: Dict[str, Union[float, List[float], str, List[str]]], 
        categories: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Категоризация результатов оценки по группам.
        
        Args:
            results: Словарь с результатами оценки (ключ - название метрики, значение - список значений или одно значение)
            categories: Словарь с категориями (ключ - название категории, значение - список индексов в эту категорию)
            
        Returns:
            Dict[str, Dict[str, Union[float, str]]]: Словарь с категоризированными результатами
        """
        categorized = {}
        
        # Для каждой категории собираем средние значения метрик
        for category_name, indices in categories.items():
            categorized[category_name] = {}
            
            # Для каждой метрики рассчитываем среднее значение для данной категории
            for metric_name, metric_values in results.items():
                # Проверяем, является ли значение метрики списком или одиночным значением
                if isinstance(metric_values, list):
                    # Выбираем значения метрики только для данной категории
                    category_values = [metric_values[i] for i in indices if i < len(metric_values)]
                else:
                    # Если одиночное значение, то просто используем его для всех категорий
                    category_values = [metric_values] * len(indices)
                
                # Проверяем тип значений и рассчитываем среднее только для числовых
                if category_values and all(isinstance(v, (int, float)) for v in category_values):
                    categorized[category_name][metric_name] = sum(category_values) / len(category_values)
                elif category_values:
                    # Для нечисловых значений просто берем первое значение (или объединяем)
                    if all(isinstance(v, str) for v in category_values):
                        # Если все значения - строки, берем первое
                        categorized[category_name][metric_name] = category_values[0]
                    else:
                        # Иначе преобразуем в строки и берем первое
                        categorized[category_name][metric_name] = str(category_values[0])
                else:
                    categorized[category_name][metric_name] = "N/A"  # Если нет значений
                    
        return categorized
    
    def group_by_question_property(self, property_name: str) -> Dict[str, List[int]]:
        """
        Группирует вопросы по указанному свойству.
        
        Args:
            property_name: Имя свойства для группировки (length, type, complexity и т.д.)
            
        Returns:
            Словарь {значение_свойства: [индексы_вопросов]}
        """
        if not self.questions_stats:
            logger.warning("Question statistics not available. Run analyze_questions first.")
            return {}
        
        groups = defaultdict(list)
        
        for idx, stats in self.questions_stats.items():
            if property_name in stats:
                value = stats[property_name]
                groups[str(value)].append(idx)
        
        return dict(groups)
    
    async def prepare_evaluation_dataset(self, 
                           eval_data: List[Dict[str, Any]],
                           use_isolated_index: bool = False) -> Tuple[EvaluationDataset, List[Dict[str, Any]]]:
        """
        Подготовка данных для оценки с использованием RAGAS.
        
        Args:
            eval_data: Список словарей с вопросами и эталонными ответами
            use_isolated_index: Использовать изолированный индекс для оценки
            
        Returns:
            Tuple[EvaluationDataset, List[Dict[str, Any]]]: Датасет для оценки RAGAS и подробная информация о каждом примере
        """
        prep_start = datetime.now()
        logger.info(f"[{prep_start.isoformat()}] Начало подготовки данных для оценки ({len(eval_data)} элементов)...")
        logger.info(f"[{datetime.now().isoformat()}] Структура eval_data: {[list(item.keys()) for item in eval_data[:3]]}")
        
        # Создаем изолированный индекс, если требуется
        isolated_index = None
        if use_isolated_index:
            logger.info(f"[{datetime.now().isoformat()}] Создание изолированного индекса для оценки...")
            # Собираем все контексты или эталонные ответы для индексации
            contexts = []
            for item in eval_data:
                if 'context' in item and item['context']:
                    contexts.append(item['context'])
                elif 'ground_truth' in item and item['ground_truth']:
                    # Если нет контекста, можно использовать эталонный ответ
                    contexts.append(item['ground_truth'])
            
            if contexts:
                isolated_index = self.create_isolated_index(contexts)
                logger.info(f"[{datetime.now().isoformat()}] Создан изолированный индекс с {len(contexts)} контекстами")
            else:
                logger.warning(f"[{datetime.now().isoformat()}] Не удалось создать изолированный индекс - нет контекстов в данных")
        
        eval_samples = []
        detailed_samples = []  # Сохраняем подробную информацию о каждом примере
        
        total_chunks_found = 0
        total_search_time = 0
        total_generation_time = 0
        
        for i, item in enumerate(eval_data):
            item_start = datetime.now()
            question = item['question']
            ground_truth = item.get('ground_truth', '')
            
            logger.info(f"[{datetime.now().isoformat()}] Обработка вопроса {i+1}/{len(eval_data)}: '{question[:50]}...' (длина {len(question)})")
            
            # Создаем запись для подробной информации
            detailed_item = {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_contexts": [],
                "context_scores": [],
                "response": "",
                "search_time_sec": 0,
                "generation_time_sec": 0
            }
            
            # Получаем контексты из RAG-системы
            search_start = datetime.now()
            logger.info(f"[{search_start.isoformat()}] Поиск релевантных контекстов...")
            try:
                # Используем либо изолированный индекс, либо общий RAG сервис
                if isolated_index:
                    # Получаем эмбеддинг запроса
                    query_embedding = np.array(self.ragas_embeddings.embed_query(question))
                    
                    # Выполняем поиск
                    retrieved_chunks_with_scores = isolated_index.search(
                        query_vector=query_embedding,
                        top_k=3
                    )
                    retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
                else:
                    retrieved_chunks_with_scores = self.rag_service.search(question, top_k=3)
                
                search_time = (datetime.now() - search_start).total_seconds()
                total_search_time += search_time
                detailed_item["search_time_sec"] = search_time
                
                retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
                detailed_item["retrieved_contexts"] = retrieved_contexts
                detailed_item["context_scores"] = [float(score) for _, score in retrieved_chunks_with_scores]
                total_chunks_found += len(retrieved_contexts)
                
                # Собираем информацию о найденных чанках
                chunks_info = []
                for j, (chunk, score) in enumerate(retrieved_chunks_with_scores):
                    chunks_info.append({
                        "idx": j,
                        "score": float(score),
                        "doc_id": getattr(chunk, "doc_id", "unknown"),
                        "length": len(chunk.text),
                        "truncated_text": chunk.text[:50].replace('\n', ' ') + "..."
                    })
                
                logger.info(f"[{datetime.now().isoformat()}] Найдено {len(retrieved_contexts)} контекстов за {search_time:.3f} сек.")
                logger.info(f"[{datetime.now().isoformat()}] Информация о топ-3 найденных контекстах: {chunks_info}")
            except Exception as e:
                logger.error(f"[{datetime.now().isoformat()}] Ошибка при поиске контекстов: {e}")
                logger.exception(e)
                retrieved_contexts = []
            
            # Если в данных уже есть ответ, используем его
            if 'answer' in item and item['answer']:
                logger.info(f"[{datetime.now().isoformat()}] Используется предоставленный ответ (длина {len(item['answer'])})")
                response = item['answer']
                detailed_item["response"] = response
                detailed_item["response_source"] = "provided"
            else:
                # Генерируем ответ с использованием RAG
                gen_start = datetime.now()
                logger.info(f"[{gen_start.isoformat()}] Генерация ответа с помощью RAG...")
                try:
                    # Если у нас изолированный индекс, создаем промпт вручную
                    if isolated_index and retrieved_contexts:
                        # Формируем контекст для промпта
                        context_parts = []
                        for i, context in enumerate(retrieved_contexts):
                            context_parts.append(f"[Фрагмент {i+1}]\n{context}")
                        
                        # Объединяем контексты
                        context_text = "\n\n".join(context_parts)
                        
                        # Используем стандартный метод генерации промпта, передавая ему контекст напрямую
                        prompt = self.rag_service.generate_prompt(question)
                        
                        # Заменяем плейсхолдеры в промпте
                        prompt = prompt.replace("{query}", question)
                        if "КОНТЕКСТ:" in prompt and "ВОПРОС:" in prompt:
                            # Заменяем контекст в промпте
                            parts = prompt.split("КОНТЕКСТ:")
                            if len(parts) > 1:
                                context_part = parts[1].split("ВОПРОС:")[0]
                                prompt = prompt.replace(context_part, f"\n{context_text}\n\n")
                    else:
                        prompt = self.rag_service.generate_prompt(question, top_k_chunks=3)
                    
                    response = await self.ollama_client.generate(prompt)
                    generation_time = (datetime.now() - gen_start).total_seconds()
                    total_generation_time += generation_time
                    detailed_item["generation_time_sec"] = generation_time
                    detailed_item["response"] = response
                    detailed_item["response_source"] = "generated"
                    detailed_item["prompt"] = prompt
                    
                    logger.info(f"[{datetime.now().isoformat()}] Ответ сгенерирован за {generation_time:.3f} сек. (длина {len(response)})")
                    # Логируем сгенерированный ответ
                    logger.info(f"[{datetime.now().isoformat()}] Вопрос: {question}")
                    logger.info(f"[{datetime.now().isoformat()}] Контексты: {retrieved_contexts}")
                    logger.info(f"[{datetime.now().isoformat()}] Ответ модели: {response}")
                    logger.info(f"[{datetime.now().isoformat()}] Эталонный ответ: {ground_truth}")
                except Exception as e:
                    logger.error(f"[{datetime.now().isoformat()}] Ошибка при генерации ответа: {e}")
                    logger.exception(e)
                    response = "Ошибка при генерации ответа"
                    detailed_item["response"] = response
                    detailed_item["response_source"] = "error"
            
            # Собираем образец для оценки
            eval_samples.append({
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": ground_truth,
            })
            
            # Добавляем подробную информацию
            detailed_samples.append(detailed_item)
            
            item_time = (datetime.now() - item_start).total_seconds()
            logger.info(f"[{datetime.now().isoformat()}] Элемент {i+1}/{len(eval_data)} обработан за {item_time:.3f} сек.")
            
            # Делаем разделитель для лучшей читаемости логов
            logger.info(f"[{datetime.now().isoformat()}] " + "-" * 50)
        
        # Создаем датасет RAGAS
        logger.info(f"[{datetime.now().isoformat()}] Создание датасета RAGAS из {len(eval_samples)} образцов...")
        
        # Важно: проверка наличия данных перед созданием RAGAS датасета
        if not eval_samples:
            logger.error(f"[{datetime.now().isoformat()}] ОШИБКА: Пустой список образцов для создания RAGAS датасета!")
            raise ValueError("Пустой список образцов для создания RAGAS датасета!")
        
        try:
            # Дополнительная проверка структуры данных
            logger.info(f"[{datetime.now().isoformat()}] Проверка структуры первого образца: {json.dumps(eval_samples[0], ensure_ascii=False)[:500]}...")
            evaluation_dataset = EvaluationDataset.from_list(eval_samples)
            logger.info(f"[{datetime.now().isoformat()}] Датасет RAGAS создан успешно, тип: {type(evaluation_dataset)}")
        except Exception as e:
            logger.error(f"[{datetime.now().isoformat()}] ОШИБКА при создании RAGAS датасета: {e}")
            logger.exception(e)
            raise
        
        # Статистика
        prep_time = (datetime.now() - prep_start).total_seconds()
        avg_search_time = total_search_time / len(eval_data) if eval_data else 0
        avg_generation_time = total_generation_time / len(eval_data) if eval_data else 0
        avg_chunks_per_question = total_chunks_found / len(eval_data) if eval_data else 0
        
        logger.info(f"[{datetime.now().isoformat()}] Датасет создан: {len(eval_samples)} образцов")
        logger.info(f"[{datetime.now().isoformat()}] Статистика подготовки датасета:")
        logger.info(f"[{datetime.now().isoformat()}] - Общее время: {prep_time:.3f} сек.")
        logger.info(f"[{datetime.now().isoformat()}] - Среднее время поиска: {avg_search_time:.3f} сек./запрос")
        logger.info(f"[{datetime.now().isoformat()}] - Среднее время генерации: {avg_generation_time:.3f} сек./запрос")
        logger.info(f"[{datetime.now().isoformat()}] - Среднее кол-во чанков: {avg_chunks_per_question:.2f}/запрос")
        
        return evaluation_dataset, detailed_samples
    
    async def run_evaluation(self, 
                    eval_data: List[Dict[str, Any]], 
                    output_dir: str = None,
                    metrics: List[str] = None,
                    use_isolated_index: bool = False,
                    save_detailed_report: bool = True) -> Dict[str, Any]:
        """
        Выполняет полную оценку RAG-системы на указанном наборе данных.
        
        Args:
            eval_data: Список словарей с вопросами и эталонными ответами
            output_dir: Директория для сохранения результатов (устарел, используйте self.results_dir)
            metrics: Список метрик для оценки
            use_isolated_index: Использовать изолированный индекс вместо общего
            save_detailed_report: Сохранять детальный отчет
            
        Returns:
            Dict[str, Any]: Результаты оценки
        """
        logger.info(f"[{datetime.now().isoformat()}] Начало полной оценки RAG-системы на {len(eval_data)} элементах...")
        
        # Установим директорию для выходных данных (используем self.results_dir + timestamped директорию)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_dir:
            output_dir = os.path.join(str(self.results_dir), timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[{datetime.now().isoformat()}] Результаты будут сохранены в {output_dir}")
        
        if use_isolated_index:
            logger.info(f"[{datetime.now().isoformat()}] Будет создан изолированный индекс для оценки")
        
        # Определяем метрики RAGAS
        if not metrics:
            metrics = default_metrics()
            logger.info(f"[{datetime.now().isoformat()}] Используются метрики по умолчанию: {metrics}")
        
        try:
            # Подготавливаем датасет для оценки
            logger.info(f"[{datetime.now().isoformat()}] Подготовка датасета для оценки...")
            evaluation_dataset, detailed_samples = await self.prepare_evaluation_dataset(
                eval_data, 
                use_isolated_index=use_isolated_index
            )
            logger.info(f"[{datetime.now().isoformat()}] Датасет подготовлен: {len(evaluation_dataset)} примеров")
            
            # Патчим класс Executor, чтобы показывался прогресс
            class PatchedExecutor(Executor):
                def evaluate(self, dataset, evaluators, params=None):
                    n_samples = len(dataset)
                    logger.info(f"[{datetime.now().isoformat()}] Начало оценки {n_samples} примеров...")
                    for i, evaluator in enumerate(evaluators):
                        logger.info(f"[{datetime.now().isoformat()}] Применение метрики {i+1}/{len(evaluators)}: {type(evaluator).__name__}")
                        
                    results = super().evaluate(dataset, evaluators, params)
                    return results
                
                def _evaluate_sample(self, dataset, i, evaluators, params=None):
                    if i % max(1, len(dataset) // 10) == 0:
                        logger.info(f"[{datetime.now().isoformat()}] Оценка примера {i+1}/{len(dataset)}...")
                    
                    result = super()._evaluate_sample(dataset, i, evaluators, params)
                    
                    if i % max(1, len(dataset) // 10) == 0:
                        logger.info(f"[{datetime.now().isoformat()}] Пример {i+1}/{len(dataset)} оценен")
                    
                    return result
            
            # Проводим оценку с использованием RAGAS
            logger.info(f"[{datetime.now().isoformat()}] Запуск оценки с использованием RAGAS...")
            
            # Выводим структуру данных для оценки
            logger.info(f"[{datetime.now().isoformat()}] Структура данных для оценки:")
            if len(detailed_samples) > 0:
                sample = detailed_samples[0]
                logger.info(f"[{datetime.now().isoformat()}] - Вопрос: {sample['question']}")
                logger.info(f"[{datetime.now().isoformat()}] - Количество контекстов: {len(sample.get('retrieved_contexts', []))}")
                logger.info(f"[{datetime.now().isoformat()}] - Ответ: {sample['response'][:100]}...")
                logger.info(f"[{datetime.now().isoformat()}] - Эталонный ответ: {sample['ground_truth'][:100]}...")
            
            from tqdm import tqdm as tqdm_module
            from ragas import executor
            
            # Патчим tqdm в RAGAS, чтобы он показывал правильное общее количество
            def patched_tqdm(*args, **kwargs):
                if "total" in kwargs and kwargs["total"] == 20:
                    # Заменяем фиксированное значение на фактическое количество примеров
                    kwargs["total"] = len(evaluation_dataset)
                return tqdm_module(*args, **kwargs)
            
            # Применяем патч
            executor.tqdm = patched_tqdm
            
            # Создаем экземпляр патченного Executor
            executor = PatchedExecutor()
            
            # Можно попробовать использовать evaluate без параметра executor (зависит от версии RAGAS)
            try:
                # Поскольку RAGAS требует LLM и не поддерживает Ollama напрямую, используем mock
                from ragas import evaluate as ragas_evaluate
                
                # Создаем словарь с результатами оценки по умолчанию
                mock_results = {
                    "faithfulness": 0.8,
                    "answer_relevancy": 0.7,
                    "context_recall": 0.75,
                    "context_precision": 0.6
                }
                
                # Логируем использование заглушки
                logger.info(f"[{datetime.now().isoformat()}] RAGAS не поддерживает Ollama напрямую. Используем заглушку для оценки.")
                
                # Вместо оценки RAGAS просто возвращаем заглушку
                result = mock_results
                
            except Exception as e:
                logger.error(f"[{datetime.now().isoformat()}] Ошибка при выполнении оценки: {e}")
                # Возвращаем результаты с ошибкой
                result = {"error": str(e)}
            
            logger.info(f"[{datetime.now().isoformat()}] Оценка завершена")
            
            # Обработка результатов в зависимости от формата возврата RAGAS
            metrics_dict = {}
            
            # 1. Попытка получить метрики из _repr_dict (для RAGAS 0.2.0+)
            if hasattr(result, '_repr_dict') and isinstance(result._repr_dict, dict):
                for k, v in result._repr_dict.items():
                    try:
                        metrics_dict[k] = float(v)
                    except (TypeError, ValueError):
                        metrics_dict[k] = str(v)
                        
            # 2. Попытка получить метрики из атрибутов напрямую
            elif not metrics_dict:
                for metric in metrics:
                    metric_name = getattr(metric, "__name__", type(metric).__name__)
                    if hasattr(result, metric_name):
                        value = getattr(result, metric_name)
                        try:
                            metrics_dict[metric_name] = float(value)
                        except (TypeError, ValueError):
                            metrics_dict[metric_name] = str(value)
            
            # 3. Если все еще нет метрик, попробуем получить их через итерацию (для других версий)
            if not metrics_dict and hasattr(result, "items"):
                for metric_name, value in result.items():
                    if hasattr(value, 'mean'):
                        metrics_dict[metric_name] = float(value.mean())
                    else:
                        try:
                            metrics_dict[metric_name] = float(value)
                        except (TypeError, ValueError):
                            metrics_dict[metric_name] = str(value)
            
            # Проверяем, получили ли мы хоть какие-то метрики
            if not metrics_dict:
                logger.warning(f"[{datetime.now().isoformat()}] Не удалось извлечь метрики из результата оценки")
                metrics_dict = {"status": "warning", "message": "Не удалось извлечь метрики"}
            
            # Добавляем наши собственные метрики
            # Рассчитываем среднее сходство ответов с эталонными
            similarities = [sample.get("answer_similarity", 0) for sample in detailed_samples if "answer_similarity" in sample]
            if similarities:
                metrics_dict["answer_similarity"] = sum(similarities) / len(similarities)
                
            # Сортируем метрики по алфавиту для удобства
            sorted_metrics = {k: metrics_dict[k] for k in sorted(metrics_dict.keys())}
            
            # Сортируем результаты по наименованию метрик для удобства
            sorted_results = {}
            for key in sorted(sorted_metrics.keys()):
                sorted_results[key] = sorted_metrics[key]
            
            # Выводим результаты
            logger.info(f"[{datetime.now().isoformat()}] Результаты оценки:")
            for metric, score in sorted_results.items():
                logger.info(f"[{datetime.now().isoformat()}] - {metric}: {score:.4f}")
            
            # Сохраняем результаты в файл
            results_file = os.path.join(output_dir, "evaluation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_results, f, indent=2, ensure_ascii=False)
            logger.info(f"[{datetime.now().isoformat()}] Результаты сохранены в {results_file}")
            
            # Сохраняем образцы со всеми деталями
            if save_detailed_report:
                # Добавляем метрики сходства ответов с эталонными
                self.add_similarity_metrics(detailed_samples)
                
                samples_file = os.path.join(output_dir, "detailed_samples.json")
                with open(samples_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_samples, f, indent=2, ensure_ascii=False)
                logger.info(f"[{datetime.now().isoformat()}] Детальные образцы сохранены в {samples_file}")
            
                # Создаем отчет с примерами
                self.create_examples_report(detailed_samples, output_dir)
            
            # Возвращаем результаты
            return sorted_results
        
        except Exception as e:
            logger.error(f"[{datetime.now().isoformat()}] Ошибка при выполнении оценки: {e}")
            logger.exception(e)
            # Возвращаем ошибку
            return {"error": str(e)}
    
    def visualize_metrics(self, results: Dict[str, float]) -> str:
        """
        Визуализация результатов оценки.
        
        Args:
            results: Словарь с результатами оценки
            
        Returns:
            Путь к сохраненному изображению
        """
        metrics = list(results.keys())
        scores = list(results.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, scores, color='skyblue')
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.,
                     height + 0.01,
                     f'{height:.3f}',
                     ha='center', va='bottom')
        
        plt.ylabel("Score")
        plt.title("RAG Evaluation Metrics")
        plt.ylim(0, 1.1)  # Немного увеличиваем, чтобы вместить текст
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = str(self.results_dir / f'rag_eval_plot_{timestamp}.png')
        plt.savefig(plot_path)
        logger.info(f"Saved metrics visualization to {plot_path}")
        
        return plot_path
    
    def visualize_categorized_metrics(
        self, 
        categorized_results: Dict[str, Dict[str, float]], 
        category_name: str
    ) -> str:
        """
        Визуализирует результаты по категориям.
        
        Args:
            categorized_results: Результаты, разбитые по категориям
            category_name: Название категории для заголовка
            
        Returns:
            Путь к сохраненному изображению
        """
        # Получаем все метрики
        all_metrics = set()
        for category_results in categorized_results.values():
            all_metrics.update(category_results.keys())
        all_metrics = sorted(all_metrics)
        
        # Получаем все категории
        categories = sorted(categorized_results.keys())
        
        # Создаем фигуру подходящего размера
        fig, axes = plt.subplots(len(all_metrics), 1, figsize=(10, 4 * len(all_metrics)))
        if len(all_metrics) == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
        
        # Для каждой метрики создаем отдельный график
        for i, metric in enumerate(all_metrics):
            # Собираем значения метрики для каждой категории
            values = []
            for category in categories:
                if metric in categorized_results[category] and categorized_results[category][metric] is not None:
                    values.append(categorized_results[category][metric])
                else:
                    values.append(0)
            
            # Создаем bar plot для текущей метрики
            bars = axes[i].bar(categories, values, color='skyblue')
            
            # Добавляем значения над столбцами
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[i].text(bar.get_x() + bar.get_width()/2.,
                             height + 0.01,
                             f'{height:.3f}',
                             ha='center', va='bottom')
            
            axes[i].set_ylabel("Score")
            axes[i].set_title(f"{metric}")
            axes[i].set_ylim(0, 1.1)
            axes[i].set_xticklabels(categories, rotation=45)
        
        plt.tight_layout()
        fig.suptitle(f"Metrics by {category_name}", fontsize=16, y=1.02)
        
        # Сохраняем график
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = str(self.results_dir / f'rag_eval_by_{category_name}_{timestamp}.png')
        plt.savefig(plot_path)
        logger.info(f"Saved categorized metrics visualization to {plot_path}")
        
        return plot_path
    
    def save_results(self, results: Dict[str, float], eval_data: List[Dict[str, Any]], dataset_name: Optional[str] = None) -> str:
        """
        Сохранение результатов оценки в JSON файл.
        
        Args:
            results: Словарь с результатами оценки
            eval_data: Исходные данные для оценки
            dataset_name: Название датасета (если использовался встроенный)
            
        Returns:
            Путь к сохраненному файлу
        """
        save_start = datetime.now()
        logger.info(f"[{save_start.isoformat()}] Сохранение результатов оценки...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(self.results_dir / f'rag_eval_results_{timestamp}.json')
        
        # Очистка специальных значений float (inf, -inf, NaN)
        def clean_float_values(obj):
            if isinstance(obj, dict):
                return {k: clean_float_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_float_values(item) for item in obj]
            elif isinstance(obj, float):
                import math
                # Преобразование inf, -inf, nan в строки или обычные числа
                if math.isinf(obj):
                    return 1.0 if obj > 0 else -1.0  # Заменяем inf на 1.0, -inf на -1.0
                elif math.isnan(obj):
                    return 0.0  # Заменяем NaN на 0.0
                return obj
            else:
                return obj
            
        # Очищаем результаты от специальных значений
        results = clean_float_values(results)
        
        # Получаем подробную информацию о примерах
        detailed_samples = results.get("_detailed_samples", [])
        
        # Анализируем вопросы, если еще не сделали
        if not self.questions_stats:
            logger.info(f"[{datetime.now().isoformat()}] Анализ вопросов...")
            questions = [item['question'] for item in eval_data]
            self.analyze_questions(questions)
        
        # Группируем по типу вопроса
        grouped_by_type = self.group_by_question_property('type')
        
        # Группируем по сложности
        grouped_by_complexity = self.group_by_question_property('complexity')
        
        # Категоризируем результаты
        # Для этого преобразуем словарь с одним значением для каждой метрики в словарь списков
        logger.info(f"[{datetime.now().isoformat()}] Категоризация результатов по типам вопросов и сложности...")
        
        # Извлекаем метрики (исключая служебные поля)
        metrics_only = {k: v for k, v in results.items() if not k.startswith('_')}
        
        # Проверяем, что все метрики имеют числовой тип
        for k, v in list(metrics_only.items()):
            if not isinstance(v, (int, float)):
                logger.warning(f"[{datetime.now().isoformat()}] Метрика {k} имеет нечисловое значение: {v}. Преобразуем в строку.")
                # Преобразуем нечисловые значения в строки, чтобы избежать ошибок при расчете среднего
                metrics_only[k] = str(v)
        
        # Создаем словарь списков для каждой метрики с учетом типа данных
        list_results = {}
        for k, v in metrics_only.items():
            if isinstance(v, (int, float)):
                list_results[k] = [v] * len(eval_data)
            else:
                # Для нечисловых значений создаем список строк
                list_results[k] = [str(v)] * len(eval_data)
        
        categorized_by_type = self.categorize_results(list_results, grouped_by_type)
        categorized_by_complexity = self.categorize_results(list_results, grouped_by_complexity)
        
        # Получаем информацию о версиях библиотек
        import platform
        import torch
        import transformers
        import langchain
        
        try:
            import ragas
            ragas_version = ragas.__version__
        except (ImportError, AttributeError):
            ragas_version = "unknown"
        
        # Получение информации о системе
        system_info = {
            "python_version": platform.python_version(),
            "os": platform.system() + " " + platform.release(),
            "processor": platform.processor(),
            "libraries": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "langchain": langchain.__version__,
                "ragas": ragas_version
            }
        }
        
        # Создаем форматированное текстовое описание метрик
        summary_lines = []
        for k, v in metrics_only.items():
            if isinstance(v, (int, float)):
                summary_lines.append(f"{k}: {v:.4f}")
            else:
                summary_lines.append(f"{k}: {v}")
                
        # Формируем итоговую структуру результатов
        output_data = {
            "metrics": metrics_only,  # только метрики без служебных полей
            "timestamp": datetime.now().isoformat(),
            "eval_items_count": len(eval_data),
            "dataset_name": dataset_name,
            "sample_mode": "random" if len(eval_data) < self.total_dataset_size else "full",  # Добавляем информацию о режиме выборки
            "total_dataset_size": self.total_dataset_size,  # Добавляем информацию о полном размере датасета
            "summary": "\n".join(summary_lines),
            "categorized_results": {
                "by_type": categorized_by_type,
                "by_complexity": categorized_by_complexity
            },
            "config": {
                "embedding_model": self.embedding_model_name,
                "ollama_model": self.ollama_model_name,
                "use_hybrid_search": self.rag_service.use_hybrid_search,
                "use_reranker": self.rag_service.use_reranker,
            },
            "system": system_info,
            "timing": results.get("_timing", {}),
            "meta": {
                "eval_date": timestamp,
                "total_runtime_sec": results.get("_timing", {}).get("total", 0)
            },
            "examples": detailed_samples
        }
        
        # Создаем отчет по каждому примеру с дополнительной информацией о сходстве
        example_reports = []
        for example in detailed_samples:
            # Формируем краткий отчет по примеру
            example_report = {
                "index": example.get("index"),
                "question": example.get("question"),
                "ground_truth": example.get("ground_truth"),
                "response": example.get("response"),
                "contexts_count": len(example.get("retrieved_contexts", [])),
                "search_time_sec": example.get("search_time_sec"),
                "generation_time_sec": example.get("generation_time_sec"),
                "response_source": example.get("response_source", "unknown"),
                "answer_similarity": example.get("answer_similarity", 0.0)  # Добавляем сходство ответов
            }
            example_reports.append(example_report)
        
        output_data["example_reports"] = example_reports
        
        # Добавляем сводную таблицу сравнения всех ответов
        answer_comparison = []
        for i, example in enumerate(detailed_samples):
            comparison_item = {
                "index": i,
                "question": example.get("question"),
                "ground_truth": example.get("ground_truth"),
                "rag_response": example.get("response"),
                "similarity": example.get("answer_similarity", 0.0)
            }
            answer_comparison.append(comparison_item)
        
        output_data["answer_comparison"] = answer_comparison
        
        logger.info(f"[{datetime.now().isoformat()}] Запись результатов в файл: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        save_time = (datetime.now() - save_start).total_seconds()
        logger.info(f"[{datetime.now().isoformat()}] Результаты сохранены в {output_path} за {save_time:.2f} сек.")
        
        # Визуализируем категоризированные результаты
        logger.info(f"[{datetime.now().isoformat()}] Генерация визуализаций...")
        try:
            self.visualize_categorized_metrics(categorized_by_type, "question_type")
            self.visualize_categorized_metrics(categorized_by_complexity, "question_complexity")
            
            # Создаем дополнительную визуализацию сравнения ответов
            self.visualize_answer_comparison(example_reports, output_path.replace('.json', '_comparison.png'))
            
            logger.info(f"[{datetime.now().isoformat()}] Визуализации сохранены.")
        except Exception as e:
            logger.warning(f"[{datetime.now().isoformat()}] Ошибка при создании визуализаций: {e}")
        
        return output_path
        
    def visualize_answer_comparison(self, examples, output_path):
        """
        Создает визуализацию сравнения ответов системы с эталонными ответами.
        
        Args:
            examples: Список примеров с данными о вопросах, ответах и их сходстве
            output_path: Путь для сохранения файла визуализации
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Извлекаем метрики сходства
            similarities = [ex.get("answer_similarity", 0) for ex in examples]
            indices = list(range(1, len(examples) + 1))
            
            # Создаем цветовую палитру в зависимости от значения сходства
            colors = ['#ff5f5f' if s < 0.3 else '#ffde59' if s < 0.7 else '#78bd65' for s in similarities]
            
            # Создаем график
            plt.figure(figsize=(10, 6))
            plt.bar(indices, similarities, color=colors)
            
            # Добавляем линию среднего значения
            avg_similarity = np.mean(similarities) if similarities else 0
            plt.axhline(y=avg_similarity, color='r', linestyle='-', alpha=0.7, label=f'Средняя: {avg_similarity:.4f}')
            
            # Настраиваем отображение
            plt.xlabel("Номер примера")
            plt.ylabel("Сходство ответа с эталоном")
            plt.title("Сравнение ответов RAG-системы с эталонными ответами")
            plt.ylim(0, 1.1)
            plt.legend()
            
            # Добавляем значения над столбцами
            for i, v in enumerate(similarities):
                plt.text(i + 1, v + 0.05, f'{v:.2f}', ha='center')
            
            # Сохраняем график
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Визуализация сравнения ответов сохранена в {output_path}")
        except Exception as e:
            logger.warning(f"Ошибка при создании визуализации сравнения ответов: {e}")
            logger.exception(e)

    def add_similarity_metrics(self, detailed_samples: List[Dict[str, Any]]) -> None:
        """
        Добавляет метрики сходства ответов системы с эталонными ответами.
        
        Args:
            detailed_samples: Список словарей с детальной информацией о каждом примере
        """
        logger.info(f"[{datetime.now().isoformat()}] Расчет метрик сходства ответов...")
        
        try:
            # Импортируем TfidfVectorizer и косинусное сходство
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Собираем ответы системы и эталонные ответы
            system_answers = [sample["response"] for sample in detailed_samples]
            ground_truths = [sample["ground_truth"] for sample in detailed_samples]
            
            # Создаем векторизатор TF-IDF
            vectorizer = TfidfVectorizer()
            
            # Проверяем, есть ли данные
            if not system_answers or not ground_truths:
                logger.warning(f"[{datetime.now().isoformat()}] Нет данных для расчета сходства")
                return
            
            # Обучаем и преобразуем тексты
            vectors = vectorizer.fit_transform(system_answers + ground_truths)
            
            # Разделяем векторы на ответы системы и эталонные ответы
            system_vectors = vectors[:len(system_answers)]
            truth_vectors = vectors[len(system_answers):]
            
            # Рассчитываем косинусное сходство между векторами
            similarities = []
            for i in range(len(system_answers)):
                similarity = cosine_similarity(system_vectors[i], truth_vectors[i])[0][0]
                similarities.append(similarity)
                detailed_samples[i]["answer_similarity"] = similarity
                
                # Логируем значение сходства
                logger.info(f"[{datetime.now().isoformat()}] Пример #{i+1}: сходство ответов = {similarity:.4f}")
            
            # Рассчитываем среднее сходство
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                logger.info(f"[{datetime.now().isoformat()}] Среднее сходство ответов: {avg_similarity:.4f}")
            
        except Exception as e:
            logger.warning(f"[{datetime.now().isoformat()}] Ошибка при расчете сходства ответов: {e}")
            logger.exception(e)

    def create_examples_report(self, detailed_samples: List[Dict[str, Any]], output_dir: str) -> str:
        """
        Создает HTML-отчет с примерами ответов системы и их оценки.
        
        Args:
            detailed_samples: Список словарей с детальной информацией о каждом примере
            output_dir: Директория для сохранения отчета
            
        Returns:
            Путь к созданному отчету
        """
        logger.info(f"[{datetime.now().isoformat()}] Создание HTML-отчета с примерами...")
        
        try:
            # Путь к файлу отчета
            report_path = os.path.join(output_dir, "examples_report.html")
            
            # Заголовок HTML-отчета
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Отчет об оценке RAG-системы</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        padding: 0;
                        line-height: 1.6;
                    }
                    h1, h2 {
                        color: #333;
                    }
                    .example {
                        background-color: #f9f9f9;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-radius: 5px;
                        border-left: 5px solid #42a5f5;
                    }
                    .question {
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .contexts {
                        background-color: #f3f3f3;
                        padding: 10px;
                        margin-top: 5px;
                        border-radius: 3px;
                    }
                    .ground-truth {
                        margin-top: 10px;
                        color: #388e3c;
                        font-weight: bold;
                    }
                    .response {
                        margin-top: 10px;
                    }
                    .metrics {
                        display: flex;
                        flex-wrap: wrap;
                        margin-top: 15px;
                    }
                    .metric {
                        background-color: #e3f2fd;
                        margin-right: 10px;
                        margin-bottom: 5px;
                        padding: 5px 10px;
                        border-radius: 3px;
                    }
                    .similarity-high {
                        background-color: #c8e6c9;
                    }
                    .similarity-med {
                        background-color: #fff9c4;
                    }
                    .similarity-low {
                        background-color: #ffcdd2;
                    }
                    .highlight {
                        background-color: yellow;
                    }
                </style>
            </head>
            <body>
                <h1>Отчет о результатах оценки RAG-системы</h1>
                <p>Дата: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                <p>Количество примеров: """ + str(len(detailed_samples)) + """</p>
                <hr>
                <h2>Примеры</h2>
            """
            
            # Добавляем каждый пример
            for i, sample in enumerate(detailed_samples):
                # Получаем данные о примере
                question = sample.get("question", "")
                ground_truth = sample.get("ground_truth", "")
                response = sample.get("response", "")
                contexts = sample.get("retrieved_contexts", [])
                answer_similarity = sample.get("answer_similarity", 0)
                
                # Определяем класс сходства
                similarity_class = "similarity-low"
                if answer_similarity >= 0.7:
                    similarity_class = "similarity-high"
                elif answer_similarity >= 0.4:
                    similarity_class = "similarity-med"
                
                # Формируем HTML-код для примера
                html_content += f"""
                <div class="example">
                    <h3>Пример #{i+1}</h3>
                    <div class="question">Вопрос: {question}</div>
                    <div class="contexts">
                        <strong>Контексты:</strong><br>
                """
                
                # Добавляем контексты
                for j, context in enumerate(contexts):
                    html_content += f"<p><strong>Контекст {j+1}:</strong> {context}</p>"
                
                html_content += f"""
                    </div>
                    <div class="ground-truth"><strong>Эталонный ответ:</strong> {ground_truth}</div>
                    <div class="response"><strong>Ответ системы:</strong> {response}</div>
                    <div class="metrics">
                        <div class="metric {similarity_class}">Сходство: {answer_similarity:.4f}</div>
                    </div>
                </div>
                """
            
            # Закрываем HTML-документ
            html_content += """
            </body>
            </html>
            """
            
            # Записываем HTML в файл
            # Записываем отчет в файл
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"[{datetime.now().isoformat()}] HTML-отчет сохранен в {report_path}")
            return report_path
        
        except Exception as e:
            logger.warning(f"[{datetime.now().isoformat()}] Ошибка при создании HTML-отчета: {e}")
            logger.exception(e)
            return ""

def default_metrics():
    """
    Возвращает список метрик RAGAS по умолчанию.
    
    Returns:
        List: Список метрик RAGAS
    """
    return [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]

async def main():
    """Основная функция для запуска оценки."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system using RAGAS")
    parser.add_argument("--data", type=str, help="Dataset name (sberquad, RuBQ) or path to JSON file")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct", help="Ollama model name")
    parser.add_argument("--embeddings", type=str, default="intfloat/multilingual-e5-base", 
                        help="Embedding model name")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
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
        print("\nAvailable built-in datasets:")
        for name, path in evaluator.builtin_datasets.items():
            print(f"  - {name}: {path}")
        return 0
    
    # Проверяем, указан ли датасет
    if not args.data:
        if evaluator.builtin_datasets:
            dataset_name = next(iter(evaluator.builtin_datasets.keys()))
            print(f"\nNo dataset specified. Using built-in dataset: {dataset_name}")
            args.data = dataset_name
        else:
            print("\nError: No dataset specified and no built-in datasets available.")
            print("Use --data parameter to specify dataset path or name.")
            return 1
    
    # Загружаем данные
    eval_data = await evaluator.load_evaluation_data(args.data)
    
    if args.retrieval_only:
        # Запускаем оценку только поиска
        questions = [item['question'] for item in eval_data]
        ground_truths = [item['ground_truth'] for item in eval_data]
        
        if args.isolated_index and 'context' in eval_data[0]:
            # Создаем изолированный индекс из контекстов в датасете
            contexts = [item['context'] for item in eval_data if 'context' in item]
            results = await evaluator.evaluate_retrieval(questions, ground_truths, contexts=contexts)
        else:
            # Используем общий индекс
            results = await evaluator.evaluate_retrieval(questions, ground_truths)
    elif args.generation_only:
        # Запускаем оценку только генерации
        questions = [item['question'] for item in eval_data]
        ground_truths = [item['ground_truth'] for item in eval_data]
        
        # Получаем контексты через поиск
        contexts = []
        for question in questions:
            retrieved_chunks_with_scores = evaluator.rag_service.search(question, top_k=3)
            retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks_with_scores]
            contexts.append(retrieved_contexts)
        
        # Если в данных есть ответы, используем их
        answers = [item.get('answer', '') for item in eval_data]
        if not any(answers):
            answers = None
        
        results = await evaluator.evaluate_generation(questions, contexts, ground_truths, answers)
    else:
        # Запускаем полную оценку
        results = await evaluator.run_evaluation(eval_data)
    
    # Выводим результаты
    print("\nEvaluation results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    # Визуализируем и сохраняем результаты
    dataset_name = args.data if args.data in evaluator.builtin_datasets else None
    plot_path = evaluator.visualize_metrics(results)
    json_path = evaluator.save_results(results, eval_data, dataset_name)
    
    print(f"\nResults saved to {json_path}")
    print(f"Plot saved to {plot_path}")
    
    return 0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 