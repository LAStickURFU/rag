# metrics.py
import os
os.environ["RAGAS_DISABLE_TELEMETRY"] = "1"

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate as ragas_evaluate
from datasets import Dataset
import logging
import warnings
import traceback
from openai import OpenAI
from sklearn.metrics import accuracy_score
from difflib import SequenceMatcher

# Логгер для метрик
logger = logging.getLogger("evaluate.metrics")

# Подготовка моделей
try:
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    sentence_model = SentenceTransformer(model_name)
    logger.info(f"Загружена модель SentenceTransformer: {model_name}")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    logger.info("Загружена модель CrossEncoder: cross-encoder/ms-marco-MiniLM-L-12-v2")
except Exception as e:
    logger.error(f"Ошибка при загрузке моделей: {e}")
    sentence_model = None
    cross_encoder = None

def compute_all_metrics(question, predicted_answer, expected_answer,
                        predicted_context, expected_context, exact_match=True):
    """
    Расчёт всех метрик для оценки качества RAG-системы.
    """
    metrics = {}

    # BERTScore (семантическое сходство ответов)
    try:
        logger.debug("Начинаем вычисление BERTScore")
        if predicted_answer and expected_answer:
            P, R, F1 = bert_score([predicted_answer], [expected_answer], lang="ru")
            metrics["bertscore"] = round(float(F1[0]), 4)
        else:
            logger.warning("Пустой ответ или ожидаемый ответ для BERTScore")
            metrics["bertscore"] = None
    except Exception as e:
        logger.error(f"Ошибка при вычислении BERTScore: {e}")
        metrics["bertscore"] = None

    # Оценка перекрытия контекста (ручной exact_match vs fuzzy)
    if expected_context:
        try:
            matches, retrieved_len, relevant_len = compute_overlap(
                predicted_context, expected_context, exact_match
            )
            recall = matches / relevant_len if relevant_len > 0 else 0.0
            precision = matches / retrieved_len if retrieved_len > 0 else 0.0
            metrics["context_recall_manual"] = round(recall, 4)
            metrics["context_precision_manual"] = round(precision, 4)
        except Exception as e:
            logger.error(f"Ошибка при вычислении перекрытия контекста: {e}")
            metrics["context_recall_manual"] = None
            metrics["context_precision_manual"] = None
    else:
        metrics["context_recall_manual"] = None
        metrics["context_precision_manual"] = None

    # Семантическая достоверность ответа относительно контекста
    try:
        sem_faith = simple_faithfulness(predicted_answer, predicted_context)
        metrics["faithfulness_semantic"] = round(float(sem_faith), 4)
    except Exception as e:
        logger.error(f"Ошибка при вычислении семантической верности: {e}")
        metrics["faithfulness_semantic"] = None

    # RAGAS: answer_relevancy и faithfulness
    try:
        # Создаем формат данных, поддерживаемый текущей версией RAGAS
        ragas_data = {
            "question": [question],
            "answer": [predicted_answer],
            "contexts": [[c for c in predicted_context]],
            "ground_truths": [[expected_answer]]
        }
        
        # Создаем датасет из словаря с нужной структурой
        ragas_input = Dataset.from_dict(ragas_data)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ragas_result = ragas_evaluate(
                ragas_input,
                metrics=[answer_relevancy, faithfulness]
            )
        
        scores = ragas_result.scores[0]
        if scores:
            metrics["ragas_answer_relevance"] = round(float(scores.get("answer_relevancy", 0.0)), 4)
            metrics["ragas_faithfulness"] = round(float(scores.get("faithfulness", 0.0)), 4)
        else:
            logger.warning("Пустой словарь .scores у ragas_result")
            metrics["ragas_answer_relevance"] = None
            metrics["ragas_faithfulness"] = None
    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик RAGAS: {e}")
        logger.error(traceback.format_exc())
        metrics["ragas_answer_relevance"] = None
        metrics["ragas_faithfulness"] = None

    # --- Добавленные новые метрики ---

    # 1) exact_match: строгое совпадение предсказанного и эталонного ответа
    try:
        em = globals()['exact_match']([predicted_answer], [expected_answer])[0]
        metrics["exact_match"] = int(em)
    except Exception as e:
        logger.error(f"Ошибка при вычислении exact_match: {e}")
        metrics["exact_match"] = None

    # 2) answer_correctness: частичная, fuzzy-оценка совпадения ответов
    try:
        ac = answer_correctness([predicted_answer], [expected_answer])[0]
        metrics["answer_correctness"] = round(ac, 4)
    except Exception as e:
        logger.error(f"Ошибка при вычислении answer_correctness: {e}")
        metrics["answer_correctness"] = None

    return metrics

def simple_faithfulness(predicted_answer, predicted_context):
    """
    Семантическая faithfulness: сравниваем embedding ответа
    со всеми фрагментами предсказанного контекста.
    """
    if sentence_model is None:
        logger.warning("SentenceTransformer не загружен для семантического перекрытия")
        return 0.0
    a_embedding = sentence_model.encode(["passage:" + predicted_answer])[0]
    context_embeddings = sentence_model.encode(["passage:" + text for text in predicted_context])
    sims = [float(cosine_similarity([a_embedding], [c])[0][0]) for c in context_embeddings]
    return max(sims) if sims else 0.0

def compute_overlap(predicted_context, expected_context, exact_match):
    """
    Ручное измерение перекрытия: если exact_match=True,
    считаем точные совпадения строк; иначе батчим семантически.
    """
    if exact_match:
        matches = sum(1 for pc in predicted_context for ec in expected_context
                      if pc.strip().lower() == ec.strip().lower())
        return matches, len(predicted_context), len(expected_context)
    # Иначе – semantic overlap через SentenceTransformer
    if sentence_model is None:
        logger.warning("SentenceTransformer не загружен, невозможно вычислить overlap")
        return 0, len(predicted_context), len(expected_context)
    thr = float(os.getenv("OVERLAP_THRESHOLD", 0.7))
    retrieved_embeds = sentence_model.encode(["passage:" + t for t in predicted_context])
    relevant_embeds = sentence_model.encode(["passage:" + t for t in expected_context])
    match_count = 0
    for r in retrieved_embeds:
        sims = util.cos_sim(r, relevant_embeds)
        if max(sims[0]) >= thr:
            match_count += 1
    return match_count, len(predicted_context), len(expected_context)

def exact_match(predictions, references):
    """Check if the predicted and reference answers are exactly the same (case-insensitive)."""
    return [int(p.strip().lower() == r.strip().lower())
            for p, r in zip(predictions, references)]

def answer_correctness(predictions, references):
    """A fuzzy comparison metric using SequenceMatcher for partial correctness."""
    return [SequenceMatcher(None,
                            p.strip().lower(),
                            r.strip().lower()).ratio()
            for p, r in zip(predictions, references)]