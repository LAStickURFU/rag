"""
Модуль для логирования и анализа случаев с низкой метрикой answer_similarity.
"""

import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluationLogger:
    """Логирование и анализ проблемных случаев в RAG-системе."""
    
    def __init__(self, log_dir: str = "app/evaluation/error_logs"):
        """
        Инициализация логгера оценки RAG.
        
        Args:
            log_dir: Директория для логов ошибок
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Путь к журналу ошибок низкого similarity
        self.low_similarity_path = os.path.join(log_dir, "low_similarity_cases.jsonl")
        self.monthly_log_path = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y_%m')}.jsonl")
        
        # Метрики отслеживания
        self.metrics_history = {
            "answer_similarity": [],
            "context_precision": [],
            "context_recall": [],
            "faithfulness": []
        }
        
        # Кэш хэшей для предотвращения дублирования логов
        self._logged_hashes: Set[str] = set()
        self._load_existing_hashes()
    
    def _load_existing_hashes(self):
        """Загружает хэши существующих логов для предотвращения дубликатов."""
        try:
            if os.path.exists(self.low_similarity_path):
                with open(self.low_similarity_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            # Создаем хэш на основе вопроса и эталонного ответа
                            question = entry.get("question", "")
                            ground_truth = entry.get("ground_truth", "")
                            hash_key = f"{question}::{ground_truth}"
                            self._logged_hashes.add(hash_key)
            logger.info(f"Loaded {len(self._logged_hashes)} existing log hashes")
        except Exception as e:
            logger.error(f"Error loading existing logs: {e}")
    
    def log_low_similarity_case(self, 
                               question: str, 
                               answer: str, 
                               ground_truth: str,
                               contexts: List[str], 
                               similarity_score: float,
                               prompt: Optional[str] = None,
                               threshold: float = 0.5):
        """
        Логирует случаи с низкой метрикой answer_similarity для анализа.
        
        Args:
            question: Вопрос пользователя
            answer: Ответ модели
            ground_truth: Эталонный ответ
            contexts: Использованные контексты
            similarity_score: Значение метрики answer_similarity
            prompt: Использованный промпт (опционально)
            threshold: Порог similarity для логирования
        """
        # Логируем только проблемные случаи ниже порога
        if similarity_score >= threshold:
            return
        
        # Предотвращаем дублирование логов одного и того же вопроса
        hash_key = f"{question}::{ground_truth}"
        if hash_key in self._logged_hashes:
            return
            
        # Добавляем хэш в кэш
        self._logged_hashes.add(hash_key)
            
        # Создаем запись
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "similarity_score": float(similarity_score),
            "contexts": contexts,
            "prompt": prompt if prompt else ""
        }
        
        # Записываем в JSONL формат (каждая строка - отдельный JSON)
        try:
            with open(self.low_similarity_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
            # Также записываем в ежемесячный лог
            with open(self.monthly_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
            logger.info(f"Logged low similarity case ({similarity_score:.4f}): {question}")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    def track_metric(self, metric_name: str, value: float):
        """
        Отслеживает значение метрики во времени.
        
        Args:
            metric_name: Название метрики
            value: Значение метрики
        """
        if metric_name in self.metrics_history:
            self.metrics_history[metric_name].append({
                "timestamp": time.time(),
                "value": float(value)
            })
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Анализирует накопленные ошибки и выдает статистику.
        
        Returns:
            Словарь со статистикой ошибок
        """
        if not os.path.exists(self.low_similarity_path):
            return {"error_count": 0}
            
        try:
            cases = []
            with open(self.low_similarity_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
            
            if not cases:
                return {"error_count": 0}
                
            # Анализ ошибок
            avg_similarity = sum(case["similarity_score"] for case in cases) / len(cases)
            questions = [case["question"] for case in cases]
            
            # Определяем типичные шаблоны ошибок
            short_answers = [case for case in cases if len(case["answer"].split()) < 3]
            long_answers = [case for case in cases if len(case["answer"].split()) > 20]
            
            # Статистика по типам вопросов с низким similarity
            question_types = {
                "who": len([q for q in questions if q.lower().startswith("кто")]),
                "what": len([q for q in questions if q.lower().startswith(("что", "какой", "какая"))]),
                "when": len([q for q in questions if q.lower().startswith(("когда", "в каком"))]),
                "where": len([q for q in questions if q.lower().startswith(("где", "куда"))]),
                "why": len([q for q in questions if q.lower().startswith(("почему", "зачем"))]),
                "how": len([q for q in questions if q.lower().startswith(("как", "каким"))]),
            }
            
            # Анализ контекста - ищем закономерности по документам
            context_sources = []
            for case in cases:
                contexts = case.get("contexts", [])
                for ctx in contexts:
                    # Ищем источник в контексте
                    if isinstance(ctx, str) and "|" in ctx:
                        parts = ctx.split("|")
                        if len(parts) > 1:
                            source = parts[1].strip()
                            context_sources.append(source)
            
            # Считаем частоту источников
            source_frequency = {}
            for source in context_sources:
                if source in source_frequency:
                    source_frequency[source] += 1
                else:
                    source_frequency[source] = 1
            
            # Сортируем источники по частоте
            problematic_sources = sorted(
                source_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5
            
            # Статистика по словам - отличия между ответом и эталоном
            common_missing_terms = self._analyze_missing_terms(cases)
            
            return {
                "error_count": len(cases),
                "avg_similarity": avg_similarity,
                "short_answers_count": len(short_answers),
                "long_answers_count": len(long_answers),
                "question_types": question_types,
                "problematic_sources": dict(problematic_sources),
                "common_missing_terms": common_missing_terms
            }
        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            return {"error_count": 0, "error": str(e)}
    
    def _analyze_missing_terms(self, cases: List[Dict[str, Any]], top_n: int = 10) -> List[str]:
        """
        Анализирует часто пропускаемые ключевые слова.
        
        Args:
            cases: Список случаев ошибок
            top_n: Количество топовых терминов для возврата
            
        Returns:
            Список часто пропускаемых ключевых слов
        """
        term_frequency = {}
        
        for case in cases:
            answer = case.get("answer", "").lower()
            ground_truth = case.get("ground_truth", "").lower()
            
            # Разбиваем на слова
            answer_words = set(answer.split())
            truth_words = set(ground_truth.split())
            
            # Находим слова, которые есть в эталоне, но отсутствуют в ответе
            missing_words = truth_words - answer_words
            
            # Обновляем частоту
            for word in missing_words:
                if len(word) > 3:  # Игнорируем короткие слова
                    if word in term_frequency:
                        term_frequency[word] += 1
                    else:
                        term_frequency[word] = 1
        
        # Сортируем по частоте
        sorted_terms = sorted(term_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем top_n терминов
        return [term for term, _ in sorted_terms[:top_n]]
    
    def get_worst_cases(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Возвращает самые проблемные случаи с наименьшим similarity.
        
        Args:
            limit: Ограничение на количество случаев
            
        Returns:
            Список наихудших случаев
        """
        if not os.path.exists(self.low_similarity_path):
            return []
            
        try:
            cases = []
            with open(self.low_similarity_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        case = json.loads(line)
                        cases.append(case)
            
            # Сортируем по similarity (возрастающий порядок)
            sorted_cases = sorted(cases, key=lambda x: x["similarity_score"])
            
            # Возвращаем ограниченное количество наихудших случаев
            worst_cases = sorted_cases[:limit]
            
            # Упрощаем данные для вывода
            simplified_cases = []
            for case in worst_cases:
                simplified_cases.append({
                    "question": case["question"],
                    "answer": case["answer"],
                    "ground_truth": case["ground_truth"],
                    "similarity_score": case["similarity_score"]
                })
                
            return simplified_cases
        except Exception as e:
            logger.error(f"Error getting worst cases: {e}")
            return []
    
    def export_analysis(self, output_path: Optional[str] = None) -> str:
        """
        Экспортирует анализ ошибок в JSON файл.
        
        Args:
            output_path: Путь для сохранения анализа (если None, создается автоматически)
            
        Returns:
            Путь к файлу с экспортированным анализом
        """
        analysis = self.analyze_errors()
        worst_cases = self.get_worst_cases(10)
        
        export_data = {
            "analysis": analysis,
            "worst_cases": worst_cases,
            "generated_at": datetime.now().isoformat()
        }
        
        if output_path is None:
            output_path = os.path.join(self.log_dir, f"error_analysis_{datetime.now().strftime('%Y%m%d')}.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported error analysis to {output_path}")
        return output_path 