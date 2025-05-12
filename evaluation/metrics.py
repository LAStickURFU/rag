from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from ragas.metrics import answer_relevancy, faithfulness
from ragas import evaluate as ragas_evaluate
from datasets import Dataset
import logging
import os
import warnings
import traceback
from openai import OpenAI

# Настройка логирования
logger = logging.getLogger("evaluate.metrics")

# Настройка OpenAI API
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

# Инициализация модели для векторизации текста - используем E5, как в основной системе
try:
    # Используем ту же модель, что и в основной системе (из .env: EMBEDDING_MODEL)
    model_name = "intfloat/multilingual-e5-base"
    sentence_model = SentenceTransformer(model_name)
    
    # Инициализируем CrossEncoder для переранжирования, как в основной системе
    # Используем тот же model_name, что и в основной системе app/retrieval/cross_encoder_reranker.py
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    logger.info(f"Загружена модель SentenceTransformer: {model_name}")
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
    
    # Вычисляем ROUGE-L (лексическое совпадение)
    try:
        logger.debug("Начинаем вычисление ROUGE-L")
        rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l = rouge.score(expected_answer, predicted_answer)['rougeL'].fmeasure
        metrics["rouge_l"] = round(rouge_l, 4)
        logger.debug(f"ROUGE-L выполнен успешно, значение: {metrics['rouge_l']}")
    except Exception as e:
        logger.error(f"Ошибка при вычислении ROUGE-L: {e}")
        metrics["rouge_l"] = None
    
    # Вычисляем BERTScore (семантическое сходство)
    try:
        logger.debug("Начинаем вычисление BERTScore")
        if predicted_answer and expected_answer:
            P, R, F1 = bert_score([predicted_answer], [expected_answer], lang="ru")
            bertscore = float(F1[0])
            metrics["bertscore"] = round(bertscore, 4)
            logger.debug(f"BERTScore выполнен успешно, значение: {metrics['bertscore']}")
        else:
            logger.warning("Пустой ответ или ожидаемый ответ для BERTScore")
            metrics["bertscore"] = None
    except Exception as e:
        logger.error(f"Ошибка при вычислении BERTScore: {e}")
        metrics["bertscore"] = None
    
    # Вычисляем метрики перекрытия контекста
    if expected_context:
        try:
            logger.debug("Начинаем вычисление перекрытия контекста")
            matches, retrieved_len, relevant_len = compute_overlap(
                predicted_context, expected_context, exact_match
            )
            recall = matches / relevant_len if relevant_len > 0 else 0.0
            precision = matches / retrieved_len if retrieved_len > 0 else 0.0
            metrics["context_recall_manual"] = round(recall, 4)
            metrics["context_precision_manual"] = round(precision, 4)
            logger.debug(
                f"Перекрытие контекста выполнено успешно. "
                f"Recall: {metrics['context_recall_manual']}, "
                f"Precision: {metrics['context_precision_manual']}"
            )
        except Exception as e:
            logger.error(f"Ошибка при вычислении перекрытия контекста: {e}")
            metrics["context_recall_manual"] = None
            metrics["context_precision_manual"] = None
    else:
        metrics["context_recall_manual"] = None
        metrics["context_precision_manual"] = None
    
    # Вычисляем семантическую достоверность
    try:
        logger.debug("Начинаем вычисление семантической достоверности (faithfulness)")
        sem_faithfulness = simple_faithfulness(predicted_answer, predicted_context)
        metrics["faithfulness_semantic"] = round(sem_faithfulness, 4)
        logger.debug(
            f"Семантическая достоверность выполнена успешно, "
            f"значение: {metrics['faithfulness_semantic']}"
        )
    except Exception as e:
        logger.error(f"Ошибка при вычислении faithfulness_semantic: {e}")
        metrics["faithfulness_semantic"] = None

    # Метрики RAGAS
    try:
        logger.info("Начинаем вычисление метрик RAGAS")
        
        # Пропускаем RAGAS если нет ответа
        if not predicted_answer or not expected_answer:
            logger.warning("Пропуск вычисления метрик RAGAS - пустой ответ или эталон")
            raise ValueError("Пустой ответ для RAGAS")

        # Подготавливаем данные для RAGAS в нужном формате
        logger.debug("Подготовка данных для RAGAS")
        # contexts должен быть списком списков строк, а reference - строкой
        ragas_data = {
            "question": [question],
            "answer": [predicted_answer],
            "contexts": [[c for c in predicted_context]],
            "ground_truths": [[expected_answer]]  # RAGAS 0.2.15 ожидает ground_truths
        }
        
        # Создаем датасет для RAGAS и отключаем предупреждения
        logger.debug("Создание датасета из данных")
        ragas_input = Dataset.from_dict(ragas_data)
        
        # Включаем лог с деталями структуры данных
        logger.info(f"RAGAS data: {ragas_data}")
        
        # Логирование типов метрик
        logger.debug("Метрики RAGAS: [answer_relevancy, faithfulness]")
        
        # Запускаем оценку RAGAS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                logger.debug("Запуск evaluate() с RAGAS метриками")
                ragas_result = ragas_evaluate(
                    ragas_input,
                    metrics=[answer_relevancy, faithfulness],
                )
                
                logger.debug(f"Результат RAGAS: {ragas_result}")
                
                # Выводим детальную информацию о структуре результата
                logger.debug(f"Тип результата RAGAS: {type(ragas_result)}")
                logger.debug(
                    f"Ключи в результате RAGAS: "
                    f"{ragas_result.keys() if hasattr(ragas_result, 'keys') else 'нет ключей'}"
                )
                
                # Обеспечиваем безопасный доступ к данным результата
                try:
                    # Собираем результаты RAGAS - исправляем способ получения значений
                    answer_rel_value = ragas_result.get("answer_relevancy", None)
                    if answer_rel_value is not None:
                        metrics["ragas_answer_relevance"] = round(float(answer_rel_value), 4)
                        logger.debug(
                            f"RAGAS answer_relevance: {metrics['ragas_answer_relevance']}"
                        )
                    else:
                        logger.warning(
                            "Ключ 'answer_relevancy' отсутствует в результате RAGAS"
                        )
                    
                    faith_value = ragas_result.get("faithfulness", None)
                    if faith_value is not None:
                        metrics["ragas_faithfulness"] = round(float(faith_value), 4)
                        logger.debug(
                            f"RAGAS faithfulness: {metrics['ragas_faithfulness']}"
                        )
                    else:
                        logger.warning(
                            "Ключ 'faithfulness' отсутствует в результате RAGAS"
                        )
                    
                    logger.info("Вычисление метрик RAGAS завершено успешно")
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(
                        f"Ошибка при извлечении значений из результата RAGAS: {e}"
                    )
                    logger.error(f"Полный результат RAGAS: {ragas_result}")
                    metrics.update({
                        "ragas_answer_relevance": None,
                        "ragas_faithfulness": None
                    })
                    
            except Exception as e:
                logger.error(f"Внутренняя ошибка при вычислении метрик RAGAS: {e}")
                logger.error(traceback.format_exc())
                metrics.update({
                    "ragas_answer_relevance": None,
                    "ragas_faithfulness": None
                })
    except Exception as e:
        logger.error(f"Ошибка при вычислении метрик RAGAS: {e}")
        logger.error(traceback.format_exc())
        metrics.update({
            "ragas_answer_relevance": None,
            "ragas_faithfulness": None
        })
    
    return metrics


def simple_faithfulness(predicted_answer, predicted_context):
    """Простая метрика достоверности на основе косинусного сходства."""
    if not sentence_model:
        logger.warning(
            "SentenceTransformer не загружен, невозможно вычислить faithfulness"
        )
        return 0.0
        
    context_text = " ".join(predicted_context)
    if not context_text or not predicted_answer:
        return 0.0
    
    # Используем префикс 'passage:' для контекста и ответа
    # в соответствии с конфигурацией основной системы
    a_embedding = sentence_model.encode(["passage:" + predicted_answer])[0]
    c_embedding = sentence_model.encode(["passage:" + context_text])[0]
    similarity = cosine_similarity([a_embedding], [c_embedding])[0][0]
    return float(similarity)


def compute_overlap(predicted_context, expected_context, exact_match=True, 
                    threshold=0.85):
    """Вычисляет перекрытие между предсказанным и ожидаемым контекстом."""
    if not predicted_context or not expected_context:
        return 0, max(len(predicted_context), 1), max(len(expected_context), 1)
        
    if exact_match:
        # Точное совпадение текстов
        intersection = set(predicted_context).intersection(set(expected_context))
        return len(intersection), len(predicted_context), len(expected_context)
    else:
        # Семантическое совпадение с использованием E5 и префиксов
        if not sentence_model:
            logger.warning(
                "SentenceTransformer не загружен для семантического перекрытия"
            )
            return 0, len(predicted_context), len(expected_context)
        
        # Используем префикс 'passage:' как в основной системе    
        retrieved_embeds = sentence_model.encode([
            "passage:" + text for text in predicted_context
        ])
        relevant_embeds = sentence_model.encode([
            "passage:" + text for text in expected_context
        ])
        match_count = 0
        for r_embed in retrieved_embeds:
            similarities = util.cos_sim(r_embed, relevant_embeds)
            if max(similarities[0]) >= threshold:
                match_count += 1
        return match_count, len(predicted_context), len(expected_context)