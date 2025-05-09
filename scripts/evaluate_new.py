"""
Скрипт для оценки RAG-системы с использованием улучшенных метрик и логирования ошибок.
"""

import sys
import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# Импортируем модули RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.rag import RAGService
from app.evaluation.logger import RAGEvaluationLogger
from app.utils.metrics_wrapper import track_latency, track_quality
from app.monitoring.metrics import get_metrics_instance

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Инициализируем систему метрик
        self.metrics_tracker = get_metrics_instance()
        
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
                return json.load(f)
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
            from sentence_transformers import SentenceTransformer, util
            
            # Загружаем модель с кэшированием, чтобы не загружать её каждый раз
            if not hasattr(self, "_similarity_model"):
                self._similarity_model = SentenceTransformer("intfloat/multilingual-e5-large")
                logger.info("Loaded similarity model for answer evaluation")
            
            # Кодируем тексты
            pred_embedding = self._similarity_model.encode(predicted, convert_to_tensor=True)
            ref_embedding = self._similarity_model.encode(reference, convert_to_tensor=True)
            
            # Вычисляем косинусное сходство
            similarity = float(util.pytorch_cos_sim(pred_embedding, ref_embedding)[0][0])
            
            return similarity
        except Exception as e:
            logger.error(f"Error computing answer similarity: {str(e)}")
            return 0.0
    
    @track_quality(metric_name="context_precision")
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
    
    @track_quality(metric_name="context_recall")
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
    
    @track_latency(phase="evaluation_example")
    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оценивает RAG-систему на одном примере.
        
        Args:
            example: Пример для оценки
            
        Returns:
            Результаты оценки для примера
        """
        question = example["question"]
        reference_answer = example["answer"]
        relevant_contexts = example.get("relevant_contexts", [])
        
        # Логируем запрос для отслеживания
        self.metrics_tracker.record_query(question)
        
        # Замеряем время поиска
        retrieval_start = time.time()
        retrieved_chunks = self.rag_service.search(question)
        retrieval_time = time.time() - retrieval_start
        
        # Записываем метрику времени поиска
        self.metrics_tracker.record_latency("retrieval", retrieval_time * 1000)
        
        # Для контекстов сохраняем только текст, чтобы их можно было сравнивать
        retrieved_contexts = [chunk.text for chunk, _ in retrieved_chunks]
        
        # Формируем промпт и генерируем ответ
        generation_start = time.time()
        prompt = self.rag_service.generate_prompt(question)
        answer = self.rag_service.generate_answer(prompt)
        generation_time = time.time() - generation_start
        
        # Записываем метрику времени генерации
        self.metrics_tracker.record_latency("generation", generation_time * 1000)
        
        # Общее время
        total_time = retrieval_time + generation_time
        self.metrics_tracker.record_latency("total", total_time * 1000)
        
        # Вычисляем метрики
        answer_similarity = self.compute_answer_similarity(answer, reference_answer)
        context_precision = self.compute_context_precision(retrieved_contexts, relevant_contexts)
        context_recall = self.compute_context_recall(retrieved_contexts, relevant_contexts)
        
        # Записываем метрики качества
        self.metrics_tracker.record_quality("answer_similarity", answer_similarity)
        self.metrics_tracker.record_quality("context_precision", context_precision)
        self.metrics_tracker.record_quality("context_recall", context_recall)
        
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
        
        # Экспортируем анализ ошибок
        analysis_file = self.error_logger.export_analysis()
        
        logger.info(f"Error analysis: {len(worst_cases)} worst cases identified")
        logger.info(f"Low similarity percentage: {aggregated_metrics['low_similarity_percentage']:.2f}%")
        logger.info(f"Error analysis exported to {analysis_file}")
        
        # Получаем общую сводку метрик из трекера
        metrics_summary = self.metrics_tracker.get_metrics_summary()
        metrics_file = os.path.join(self.output_dir, f"metrics_summary_{timestamp}.json")
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Metrics summary saved to {metrics_file}")
        
        return aggregated_metrics

def main():
    """
    Основная функция для запуска оценки.
    """
    parser = argparse.ArgumentParser(description="Evaluate RAG system with improved metrics and error logging")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, default="app/evaluation/results", 
                      help="Directory to save results")
    parser.add_argument("--model", type=str, default="mistral:7b-instruct", 
                      help="LLM model to use")
    parser.add_argument("--cache-embeddings", action="store_true", 
                      help="Use embedding cache")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="Similarity threshold for error logging")
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
        temperature=0.1,  # Низкая температура для более детерминированных ответов
        max_tokens=512,  # Ограничиваем длину ответа
        top_p=0.9,
        presence_penalty=0.2
    )
    rag_service.llm_client = ollama_client
    
    # Инициализируем оценщик
    evaluator = RAGEvaluator(
        rag_service=rag_service,
        dataset_file=args.dataset,
        output_dir=args.output_dir,
        similarity_threshold=args.threshold
    )
    
    # Запускаем оценку
    results = evaluator.run_evaluation()
    
    # Выводим агрегированные метрики
    print("\nEvaluation Results:")
    print(f"Answer Similarity: {results['answer_similarity_avg']:.4f} ± {results['answer_similarity_std']:.4f}")
    print(f"Context Precision: {results['context_precision_avg']:.4f}")
    print(f"Context Recall: {results['context_recall_avg']:.4f}")
    print(f"Retrieval Latency: {results['retrieval_latency_avg']:.2f}s")
    print(f"Generation Latency: {results['generation_latency_avg']:.2f}s")
    print(f"Total Latency: {results['total_latency_avg']:.2f}s")
    print(f"Low Similarity Cases: {results['low_similarity_percentage']:.2f}%")

if __name__ == "__main__":
    main() 