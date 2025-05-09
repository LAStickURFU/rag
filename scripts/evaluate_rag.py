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
import sys
import time
import asyncio

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

# Импорты для патчинга RAGAS Executor
import types
from functools import wraps
from tqdm.auto import tqdm
from ragas.run_config import RunConfig

# Импорты из проекта
from app.rag import RAGService, Document
from app.ollama_client import OllamaLLM, get_ollama_instance
from app.chunking.robust_chunker import RobustChunker
from app.evaluation.logger import RAGEvaluationLogger
from app.utils.metrics_wrapper import track_latency, track_quality

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

# Функция для обработки ошибок с SentenceTransformer и предоставления резервного метода
def safe_import_sentence_transformer(model_name="intfloat/multilingual-e5-large"):
    """
    Безопасно импортирует SentenceTransformer и возвращает инициализированную модель или None
    
    Args:
        model_name: Название модели для загрузки
        
    Returns:
        Модель SentenceTransformer или None в случае ошибки
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded SentenceTransformer model: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"Error importing or loading SentenceTransformer: {str(e)}")
        try:
            # Пробуем альтернативную модель
            alternative_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            logger.info(f"Trying alternative model: {alternative_model}")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(alternative_model)
            logger.info(f"Successfully loaded alternative SentenceTransformer model")
            return model
        except Exception as alt_e:
            logger.warning(f"Error loading alternative model: {str(alt_e)}")
            return None

@track_latency(phase="evaluation")
class RAGEvaluator:
    """
    Класс для оценки RAG-системы.
    """
    
    def __init__(self, 
                rag_service: RAGService,
                dataset_file: str,
                output_dir: str = "app/evaluation/results",
                similarity_threshold: float = 0.5):
        """
        Инициализирует оценщик RAG-системы.
        
        Args:
            rag_service: Экземпляр RAG-сервиса для оценки
            dataset_file: Путь к файлу с датасетом оценки
            output_dir: Директория для сохранения результатов
            similarity_threshold: Порог для логирования низкой метрики answer_similarity
        """
        self.rag_service = rag_service
        self.dataset_file = dataset_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализируем логгер ошибок
        self.error_logger = RAGEvaluationLogger(
            log_dir="app/evaluation/error_logs"
        )
        
        # Инициализируем Ollama клиент для генерации ответов
        self.ollama_client = get_ollama_instance()
        
        # Загрузка датасета
        self.dataset = self._load_dataset(dataset_file)
        logger.info(f"Loaded evaluation dataset with {len(self.dataset)} examples")
        
        # Для отслеживания метрик
        self.results = []
        self.similarity_threshold = similarity_threshold
        
        # Метрики
        self.metrics = {
            "answer_similarity": [],
            "context_precision": [],
            "context_recall": [],
            "retrieval_latency": [],
            "generation_latency": [],
            "total_latency": []
        }
        
        # Применяем патчи и исправления
        patch_ragas_executor()
        patch_ragas_evaluate()
    
    def _load_dataset(self, dataset_file: str) -> List[Dict[str, Any]]:
        """
        Загружает датасет оценки.
        
        Args:
            dataset_file: Путь к файлу с датасетом
            
        Returns:
            Список примеров для оценки
        """
        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
                
                # Проходим по датасету и обеспечиваем, чтобы у каждого примера было поле ground_truth
                for item in dataset:
                    # Если есть ground_truth, но нет answer, копируем ground_truth в answer для совместимости
                    if "ground_truth" in item and not item.get("answer"):
                        item["answer"] = item["ground_truth"]
                    # Если есть answer, но нет ground_truth, копируем answer в ground_truth
                    elif "answer" in item and not item.get("ground_truth"):
                        item["ground_truth"] = item["answer"]
                        
                return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return []
    
    @track_quality(metric_name="answer_similarity")
    def compute_answer_similarity(self, predicted: str, reference: str) -> float:
        """
        Вычисляет семантическое сходство между предсказанным ответом и эталоном.
        
        Args:
            predicted: Предсказанный ответ модели
            reference: Эталонный ответ из датасета
            
        Returns:
            Оценка сходства (0-1)
        """
        try:
            # Пытаемся использовать семантическое сходство если есть модель
            if not hasattr(self, "_similarity_model") or self._similarity_model is None:
                # Загрузка модели при первом использовании
                self._similarity_model = safe_import_sentence_transformer()
                
            # Если у нас есть рабочая модель, используем её
            if self._similarity_model is not None:
                from sentence_transformers import util
                # Кодируем тексты
                pred_embedding = self._similarity_model.encode(predicted, convert_to_tensor=True)
                ref_embedding = self._similarity_model.encode(reference, convert_to_tensor=True)
                
                # Вычисляем косинусное сходство
                similarity = float(util.pytorch_cos_sim(pred_embedding, ref_embedding)[0][0])
                return similarity
                
            # Если модель недоступна, используем текстовое сходство
            return self.compute_string_similarity(predicted, reference)
        except Exception as e:
            logger.warning(f"Failed to compute semantic similarity: {str(e)}. Using string similarity instead.")
            return self.compute_string_similarity(predicted, reference)
    
    def compute_string_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет простое текстовое сходство между двумя строками.
        Использует overlap coefficient и cosine similarity на уровне слов.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Оценка сходства (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Нормализуем тексты: приводим к нижнему регистру, убираем пунктуацию
        import re
        from collections import Counter
        
        def normalize_text(text):
            # Удаляем пунктуацию, приводим к нижнему регистру
            text = re.sub(r'[^\w\s]', '', text.lower())
            # Разбиваем на слова
            return text.split()
        
        # Нормализуем и токенизируем
        tokens1 = normalize_text(text1)
        tokens2 = normalize_text(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
            
        # Создаем множества и счетчики слов
        set1 = set(tokens1)
        set2 = set(tokens2)
        count1 = Counter(tokens1)
        count2 = Counter(tokens2)
        
        # Вычисляем overlap coefficient (отношение размера пересечения к размеру меньшего множества)
        # Это даёт хорошие результаты, когда один текст является подмножеством другого
        overlap = len(set1.intersection(set2)) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
        
        # Вычисляем косинусное сходство между векторами частот слов
        # Это даёт хорошие результаты для текстов разной длины с похожей лексикой
        dot_product = sum(count1[word] * count2[word] for word in set1.intersection(set2))
        magnitude1 = sum(count**2 for count in count1.values()) ** 0.5
        magnitude2 = sum(count**2 for count in count2.values()) ** 0.5
        
        cosine = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
        
        # Возвращаем взвешенное среднее обоих метрик
        return 0.5 * overlap + 0.5 * cosine
    
    def compute_context_precision(self, retrieved_contexts: List[str], 
                                 relevant_contexts: List[str]) -> float:
        """
        Вычисляет точность контекста (какая часть извлеченного контекста релевантна).
        
        Args:
            retrieved_contexts: Извлеченные системой контексты
            relevant_contexts: Релевантные контексты из датасета
            
        Returns:
            Точность (0-1)
        """
        if not retrieved_contexts:
            return 0.0
        
        relevant_count = sum(1 for ctx in retrieved_contexts if ctx in relevant_contexts)
        return relevant_count / len(retrieved_contexts)
    
    def compute_context_recall(self, retrieved_contexts: List[str], 
                              relevant_contexts: List[str]) -> float:
        """
        Вычисляет полноту контекста (какая часть релевантного контекста была извлечена).
        
        Args:
            retrieved_contexts: Извлеченные системой контексты
            relevant_contexts: Релевантные контексты из датасета
            
        Returns:
            Полнота (0-1)
        """
        if not relevant_contexts:
            return 1.0  # Если нет релевантных контекстов, полнота 100%
        
        relevant_count = sum(1 for ctx in retrieved_contexts if ctx in relevant_contexts)
        return relevant_count / len(relevant_contexts)
    
    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценивает RAG-систему на одном примере.
        
        Args:
            example: Пример для оценки
            
        Returns:
            Результаты оценки для примера
        """
        question = example["question"]
        # Используем ground_truth вместо answer, так как в датасетах эталонные ответы в этом поле
        reference_answer = example.get("ground_truth", example.get("answer", ""))
        
        # Получаем предопределенные релевантные контексты или создаем пустой список
        relevant_contexts = example.get("relevant_contexts", [])
        
        # Замеряем время поиска
        retrieval_start = time.time()
        retrieved_chunks = self.rag_service.search(question)
        retrieval_time = time.time() - retrieval_start
        
        # Для контекстов сохраняем только текст, чтобы их можно было сравнивать
        retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks]
        
        # Формируем промпт и генерируем ответ
        generation_start = time.time()
        prompt = self.rag_service.generate_prompt(question)
        
        # Используем синхронный метод generate_sync
        try:
            answer = self.ollama_client.generate_sync(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = f"Ошибка генерации: {str(e)}"
        
        generation_time = time.time() - generation_start
        
        # Если нет предопределенных релевантных контекстов, автоматически определяем их
        # путем поиска фрагментов, содержащих части эталонного ответа
        if not relevant_contexts and reference_answer:
            try:
                # Загружаем модель для эмбеддингов, если её еще нет
                if not hasattr(self, "_similarity_model"):
                    from sentence_transformers import SentenceTransformer
                    self._similarity_model = SentenceTransformer("intfloat/multilingual-e5-large")
                    logger.info("Loaded similarity model for context relevance detection")
                
                # Вычисляем эмбеддинги для ответа и контекстов
                ref_embedding = self._similarity_model.encode(reference_answer, convert_to_tensor=True)
                context_embeddings = self._similarity_model.encode(retrieved_contexts, convert_to_tensor=True)
                
                # Вычисляем схожесть эталонного ответа с каждым контекстом
                from sentence_transformers import util
                similarities = util.pytorch_cos_sim(ref_embedding, context_embeddings)
                
                # Считаем контекст релевантным, если его сходство с эталонным ответом выше порога
                similarity_threshold = 0.4  # Порог для определения релевантности
                for i, similarity in enumerate(similarities[0]):
                    if similarity > similarity_threshold:
                        relevant_contexts.append(retrieved_contexts[i])
                
                logger.info(f"Автоматически определены {len(relevant_contexts)} релевантных контекстов")
            except Exception as e:
                logger.warning(f"Не удалось автоматически определить релевантные контексты: {str(e)}")
                # Если ошибка, считаем все контексты релевантными, если они содержат части ответа
                for ctx in retrieved_contexts:
                    if reference_answer and any(part in ctx for part in reference_answer.split()):
                        relevant_contexts.append(ctx)
        
        # Вычисляем метрики
        answer_similarity = self.compute_answer_similarity(answer, reference_answer)
        context_precision = self.compute_context_precision(retrieved_contexts, relevant_contexts)
        context_recall = self.compute_context_recall(retrieved_contexts, relevant_contexts)
        
        # Общее время
        total_time = retrieval_time + generation_time
        
        # Логируем случаи с низким сходством ответов
        if answer_similarity < self.similarity_threshold:
            self.error_logger.log_low_similarity_case(
                question=question,
                answer=answer,
                ground_truth=reference_answer,
                contexts=retrieved_contexts,
                similarity_score=answer_similarity,
                prompt=prompt,
                threshold=self.similarity_threshold
            )
        
        # Возвращаем результаты
        return {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "relevant_contexts": relevant_contexts,
            "metrics": {
                "answer_similarity": answer_similarity,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Запускает полную оценку RAG-системы на всем датасете.
            
        Returns:
            Результаты оценки
        """
        # Начинаем оценку
        start_time = time.time()
        logger.info(f"Starting evaluation on {len(self.dataset)} examples")
        
        for i, example in enumerate(self.dataset):
            try:
                result = self.evaluate_example(example)
                self.results.append(result)
                
                # Обновляем метрики
                metrics = result["metrics"]
                self.metrics["answer_similarity"].append(metrics["answer_similarity"])
                self.metrics["context_precision"].append(metrics["context_precision"])
                self.metrics["context_recall"].append(metrics["context_recall"])
                self.metrics["retrieval_latency"].append(metrics["retrieval_time"])
                self.metrics["generation_latency"].append(metrics["generation_time"])
                self.metrics["total_latency"].append(metrics["total_time"])
                
                # Логируем прогресс
                if (i + 1) % 10 == 0 or (i + 1) == len(self.dataset):
                    logger.info(f"Processed {i + 1}/{len(self.dataset)} examples")
                
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {str(e)}")
        
        # Вычисляем агрегированные метрики
        aggregated_metrics = {
            "answer_similarity_avg": np.mean(self.metrics["answer_similarity"]),
            "answer_similarity_std": np.std(self.metrics["answer_similarity"]),
            "context_precision_avg": np.mean(self.metrics["context_precision"]),
            "context_recall_avg": np.mean(self.metrics["context_recall"]),
            "retrieval_latency_avg": np.mean(self.metrics["retrieval_latency"]),
            "generation_latency_avg": np.mean(self.metrics["generation_latency"]),
            "total_latency_avg": np.mean(self.metrics["total_latency"]),
        }
        
        # Добавляем дополнительную статистику
        low_similarity_count = sum(1 for sim in self.metrics["answer_similarity"] 
                                 if sim < self.similarity_threshold)
        aggregated_metrics["low_similarity_percentage"] = (low_similarity_count / len(self.dataset)) * 100
        
        # Сохраняем результаты
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_file,
            "examples_count": len(self.dataset),
            "aggregated_metrics": aggregated_metrics,
            "detailed_results": self.results,
            "evaluation_time": time.time() - start_time
        }
        
        # Сохраняем в файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation complete. Results saved to {result_file}")
        
        # Анализируем ошибки и экспортируем анализ
        error_analysis = self.error_logger.analyze_errors()
        worst_cases = self.error_logger.get_worst_cases(10)
        
        logger.info(f"Error analysis: {len(worst_cases)} worst cases identified")
        logger.info(f"Low similarity percentage: {aggregated_metrics['low_similarity_percentage']:.2f}%")
        
        return aggregated_metrics

def main():
    """
    Основная функция для запуска оценки.
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, default="app/evaluation/results", 
                      help="Directory to save results")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct", 
                      help="LLM model to use")
    parser.add_argument("--cache-embeddings", action="store_true", 
                      help="Use embedding cache")
    args = parser.parse_args()
    
    # Инициализируем RAG-сервис с улучшенными параметрами
    rag_service = RAGService(
        model_name="intfloat/multilingual-e5-large",  # Улучшенная модель эмбеддингов
        dense_weight=0.6,  # Новые оптимизированные веса
        reranker_weight=0.6,
        chunk_size=400,  
        chunk_overlap=200,  # Увеличенное перекрытие
        use_hybrid_search=True,
        use_reranker=True,
        use_adaptive_k=True,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Улучшенная модель cross-encoder
        cache_embeddings=args.cache_embeddings  # Используем кэширование, если указано
    )
    
    # Инициализируем OllamaLLM с новыми параметрами
    from app.ollama_client import OllamaLLM
    ollama_client = OllamaLLM(
        model_name=args.model,
        temperature=0,     # Низкая температура для более детерминированных ответов
        num_predict=512,   # Количество токенов для генерации
        top_p=0.95,        # Вероятностный порог для сэмплирования
        top_k=20,          # Ограничиваем список кандидатов для каждого токена
        stop=['<|im_end|>', '</answer>', '\n\n', '\nВОПРОС:', 'ВОПРОС:'],  # Стоп-токены
        repeat_penalty=1.15,  # Штраф за повторение
        frequency_penalty=0.05  # Штраф за частоту
    )
    rag_service.llm_client = ollama_client
    
    # Инициализируем оценщик
    evaluator = RAGEvaluator(
        rag_service=rag_service,
        dataset_file=args.dataset,
        output_dir=args.output_dir
    )
    
    # Запускаем оценку
    results = evaluator.run_evaluation()
    
    # Выводим агрегированные метрики
    print("\nEvaluation Results:")
    print(f"Answer Similarity: {results['answer_similarity_avg']:.4f}")
    print(f"Context Precision: {results['context_precision_avg']:.4f}")
    print(f"Context Recall: {results['context_recall_avg']:.4f}")
    print(f"Retrieval Latency: {results['retrieval_latency_avg']:.2f}s")
    print(f"Generation Latency: {results['generation_latency_avg']:.2f}s")
    print(f"Total Latency: {results['total_latency_avg']:.2f}s")

if __name__ == "__main__":
    main()