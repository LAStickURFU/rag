import json
import argparse
from tqdm import tqdm
import requests
from metrics import compute_all_metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
from openai import OpenAI

# Создаем директорию для логов, если она не существует
logs_dir = "logs"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("evaluate")

API_URL = "http://localhost:8000/ask"
AUTH_URL = "http://localhost:8000/token"
USERNAME = "mass_upload"
PASSWORD = "Qwerty123!"
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

def get_auth_token():
    """Получает токен авторизации для доступа к API."""
    try:
        response = requests.post(
            AUTH_URL,
            data={"username": USERNAME, "password": PASSWORD}
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            logger.error(f"Ошибка авторизации: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при запросе токена: {e}")
        return None

def load_test_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ask_question(question):
    """Отправляет вопрос в API и возвращает ответ с контекстом."""
    token = get_auth_token()
    if not token:
        logger.error("Не удалось получить токен авторизации")
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(
            API_URL, json={"question": question}, headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Получен ответ от API для вопроса: {question[:30]}...")
            
            # Проверяем наличие контекста в ответе
            context = data.get("context", [])
            if not context and data.get("relevant_chunks"):
                # Если контекст доступен в другом формате
                context = [
                    chunk.get("text", "") 
                    for chunk in data.get("relevant_chunks", [])
                ]
            
            return {
                "answer": data.get("answer", "") or data.get("response", ""),
                "context": context
            }
        else:
            logger.error(
                f"Ошибка при запросе к API: {response.status_code}, "
                f"ответ: {response.text}"
            )
            return None
    except Exception as e:
        logger.error(f"Ошибка при запросе к API: {e}")
        return None

def ask_openai_judges(question, answer, context):
    """
    Использует GPT (gpt-3.5-turbo) для оценки качества ответа по двум критериям:
    - llm_relevance — насколько ответ релевантен вопросу
    - llm_faithfulness — насколько ответ основан на предоставленном контексте
    
    Для оценки faithfulness контекст представляется в виде пронумерованного 
    списка релевантных фрагментов, что позволяет LLM лучше анализировать 
    структуру контекста и оценивать точность ответа относительно фрагментов.
    """
    # Форматируем контекст в виде пронумерованного списка
    context_formatted = "\n\n".join(
        [f"Фрагмент {i+1}:\n{text}" for i, text in enumerate(context)]
    )

    logger.debug(f"Контекст: {context_formatted}")

    prompts = {
        "llm_relevance": f"""
        Вопрос: {question}
        Ответ: {answer}

        Насколько данный ответ релевантен вопросу, независимо от контекста? 
        Ответь одним числом от 0 до 1.
        """,
        "llm_faithfulness": f"""
        Ответ: {answer}
        
        Контекст (найденные релевантные фрагменты):
        {context_formatted}

        Насколько данный ответ соответствует только предоставленному контексту, 
        без добавления лишней информации? Ответь одним числом от 0 до 1.
        """
    }

    scores = {}
    for key, prompt in prompts.items():
        try:
            # Используем API OpenAI v1
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", 
                     "content": "Ты - точный оценщик качества ответов. "
                                "Отвечай только числом от 0 до 1."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            # Извлекаем результат из структуры ответа OpenAI v1
            result = response.choices[0].message.content.strip()
            
            # Пытаемся извлечь число из ответа
            try:
                # Ищем первое число в ответе
                import re
                numbers = re.findall(r"0\.\d+|\d+\.?\d*", result)
                if numbers:
                    score = float(numbers[0])
                    # Ограничиваем значение от 0 до 1
                    score = max(0.0, min(1.0, score))
                    scores[key] = score
                else:
                    logger.warning(
                        f"Не удалось найти числовое значение в ответе LLM: {result}"
                    )
                    scores[key] = None
            except (ValueError, IndexError) as e:
                logger.error(
                    f"Ошибка при извлечении числа из ответа LLM: {e}, ответ: {result}"
                )
                scores[key] = None
        except Exception as e:
            logger.error(f"Ошибка при вызове OpenAI для {key}: {e}")
            scores[key] = None
    
    return scores

def plot_metrics_bar_chart(df, output_base):
    """Создает столбчатую диаграмму средних значений метрик."""
    means = df["mean"].dropna()
    if means.empty:
        logger.warning("Предупреждение: нет данных для построения графика метрик")
        return
    
    try:
        plt.figure(figsize=(10, 6))
        means.sort_values().plot(kind='barh')
        plt.title("Средние значения метрик")
        plt.xlabel("Значение")
        plt.tight_layout()
        plt.savefig(f"{output_base}_metrics.svg")
        plt.savefig(f"{output_base}_metrics.png")
        plt.close()
        logger.info(f"График метрик сохранен в {output_base}_metrics.png/svg")
    except Exception as e:
        logger.error(f"Ошибка при построении графика метрик: {e}")

def plot_metrics_heatmap(summary, output_base):
    """Создает тепловую карту значений метрик для каждого примера."""
    # Создаем DataFrame с данными метрик без столбца mean
    data = {k: v for k, v in summary.items() if k != "mean" and v and len(v) > 0}
    if not data:
        logger.warning("Нет данных для построения тепловой карты")
        return
    
    try:
        df = pd.DataFrame(data)
        
        # Замена None на NaN для корректного отображения
        df = df.replace({None: np.nan})
        
        # Переименование индексов для более понятного отображения
        df.index = [f"Пример {i+1}" for i in range(len(df))]
        
        plt.figure(figsize=(12, len(df)*0.8 + 2))
        sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", 
                    linewidths=.5, cbar_kws={"label": "Значение"})
        plt.title("Тепловая карта метрик по примерам")
        plt.tight_layout()
        plt.savefig(f"{output_base}_heatmap.svg")
        plt.savefig(f"{output_base}_heatmap.png")
        plt.close()
        logger.info(f"Тепловая карта сохранена в {output_base}_heatmap.png/svg")
    except Exception as e:
        logger.error(f"Ошибка при построении тепловой карты: {e}")

def plot_metrics_boxplot(summary, output_base):
    """Создает диаграмму размаха для метрик."""
    df = pd.DataFrame(
        {k: v for k, v in summary.items() if k != "mean" and v and len(v) > 0}
    )
    if df.empty:
        logger.warning("Нет данных для построения диаграммы размаха")
        return
    
    try:
        plt.figure(figsize=(12, 6))
        df.boxplot(rot=45)
        plt.title("Распределение метрик по выборке")
        plt.ylabel("Значение")
        plt.tight_layout()
        plt.savefig(f"{output_base}_boxplot.svg")
        plt.savefig(f"{output_base}_boxplot.png")
        plt.close()
        logger.info(f"Диаграмма размаха сохранена в {output_base}_boxplot.png/svg")
    except Exception as e:
        logger.error(f"Ошибка при построении диаграммы размаха: {e}")

def plot_llm_scatter(summary, output_base):
    """Создает диаграмму рассеяния для метрик LLM."""
    relevance = summary.get("llm_relevance", [])
    faithfulness = summary.get("llm_faithfulness", [])
    if (not relevance or not faithfulness or 
            len(relevance) == 0 or len(faithfulness) == 0):
        logger.warning(
            "Недостаточно данных для построения диаграммы рассеяния LLM"
        )
        return
    
    # Фильтруем None значения
    valid_pairs = [(r, f) for r, f in zip(relevance, faithfulness) 
                  if r is not None and f is not None]
    if not valid_pairs:
        logger.warning("Нет валидных пар значений для диаграммы рассеяния")
        return
    
    relevance_filtered, faithfulness_filtered = zip(*valid_pairs)
    
    try:
        plt.figure(figsize=(6, 6))
        plt.scatter(relevance_filtered, faithfulness_filtered, 
                   alpha=0.6, s=100, c='royalblue')
        
        # Добавляем подписи к точкам
        for i, (x, y) in enumerate(zip(relevance_filtered, faithfulness_filtered)):
            plt.annotate(f"Пример {i+1}", (x, y), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel("LLM Relevance")
        plt.ylabel("LLM Faithfulness")
        plt.title("Соотношение LLM-показателей")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Устанавливаем диапазон осей более информативно
        x_min = min(relevance_filtered) - 0.05 if relevance_filtered else 0
        x_max = max(relevance_filtered) + 0.05 if relevance_filtered else 1
        y_min = min(faithfulness_filtered) - 0.05 if faithfulness_filtered else 0
        y_max = max(faithfulness_filtered) + 0.05 if faithfulness_filtered else 1
        
        plt.xlim(max(0, x_min), min(1, x_max))
        plt.ylim(max(0, y_min), min(1, y_max))
        
        plt.tight_layout()
        plt.savefig(f"{output_base}_llm_scatter.svg")
        plt.savefig(f"{output_base}_llm_scatter.png")
        plt.close()
        logger.info(
            f"Диаграмма рассеяния LLM сохранена в {output_base}_llm_scatter.png/svg"
        )
    except Exception as e:
        logger.error(f"Ошибка при построении диаграммы рассеяния LLM: {e}")

def plot_metrics_grouped(summary, output_base):
    """Создает группированные визуализации метрик по категориям."""
    data = {k: v for k, v in summary.items() if k != "mean" and v and len(v) > 0}
    if not data:
        logger.warning("Нет данных для группированной визуализации")
        return
    
    try:
        df = pd.DataFrame(data)
        
        # Группируем метрики по категориям
        metric_groups = {
            "Текстовое сходство": ["rouge_l", "bertscore"],
            "Контекст": ["context_recall_manual", "context_precision_manual"],
            "Семантика": ["faithfulness_semantic"],
            "LLM оценки": ["llm_relevance", "llm_faithfulness"],
            "RAGAS": ["ragas_answer_relevance", "ragas_faithfulness"]
        }
        
        # Создаем сводную таблицу по группам метрик
        for group_name, metrics in metric_groups.items():
            group_data = df[[m for m in metrics if m in df.columns]]
            if group_data.empty:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Используем разные типы графиков в зависимости от количества примеров
            if len(group_data) <= 5:
                # Для малого количества примеров используем точечную диаграмму
                for metric in group_data.columns:
                    plt.plot(group_data.index, group_data[metric], 'o-', 
                            label=metric, markersize=8)
                    
                plt.xticks(group_data.index, [f"Пример {i+1}" for i in range(len(group_data))])
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
            else:
                # Для большего количества примеров используем виолинплот
                df_melted = pd.melt(group_data.reset_index(), 
                                    id_vars='index', 
                                    var_name='Метрика', 
                                    value_name='Значение')
                sns.violinplot(x='Метрика', y='Значение', data=df_melted)
                
            plt.title(f"Группа метрик: {group_name}")
            plt.ylim(0, 1.1)  # Устанавливаем одинаковый диапазон для всех групп
            plt.tight_layout()
            plt.savefig(f"{output_base}_group_{group_name.lower().replace(' ', '_')}.svg")
            plt.savefig(f"{output_base}_group_{group_name.lower().replace(' ', '_')}.png")
            plt.close()
            
        logger.info(f"Группированные визуализации сохранены в {output_base}_group_*.png/svg")
    except Exception as e:
        logger.error(f"Ошибка при создании группированных визуализаций: {e}")

def evaluate_one(item, exact_match):
    question = item['question']
    expected_answer = item['expected_answer']
    expected_context = item.get('expected_context', [])

    logger.info(f"Оценка вопроса: {question}")
    prediction = ask_question(question)
    if prediction is None:
        logger.warning(f"Не удалось получить ответ для вопроса: {question}")
        return None, {}

    predicted_answer = prediction['answer']
    predicted_context = prediction['context']

    metrics = compute_all_metrics(
        question=question,
        predicted_answer=predicted_answer,
        expected_answer=expected_answer,
        predicted_context=predicted_context,
        expected_context=expected_context,
        exact_match=exact_match
    )

    llm_scores = ask_openai_judges(question, predicted_answer, predicted_context)
    metrics.update(llm_scores)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "predicted_answer": predicted_answer,
        "metrics": metrics
    }, metrics

def evaluate_all(dataset, output_path, exact_match):
    results = []
    summary = {}
    question_map = {}

    for item in tqdm(dataset, desc="Evaluating..."):
        result, metrics = evaluate_one(item, exact_match)
        if result is not None:
            results.append(result)
            for key, value in metrics.items():
                summary.setdefault(key, []).append(value)
                question_map.setdefault(key, []).append(result['question'])

    # Создаем директорию для результатов, если она не существует
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Создана директория для результатов: {output_dir}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nСохранены результаты оценки в {output_path}")

    logger.info("\n--- Сводка по метрикам ---")
    df = pd.DataFrame.from_dict(summary, orient='index')
    df["mean"] = df.apply(
        lambda row: (
            round(
                sum(v for v in row if v is not None) / 
                len([v for v in row if v is not None]), 4
            ) 
            if any(v is not None for v in row) else None
        ), 
        axis=1
    )
    logger.info(df[["mean"]].to_string())

    csv_path = output_path.replace(".json", "_summary.csv")
    df.to_csv(csv_path)
    logger.info(f"\nСохранена сводка в {csv_path}")

    output_base = output_path.replace(".json", "")
    plot_metrics_bar_chart(df, output_base)
    plot_metrics_boxplot(summary, output_base)
    plot_llm_scatter(summary, output_base)
    plot_metrics_heatmap(summary, output_base)
    plot_metrics_grouped(summary, output_base)
    
    logger.info(f"Сохранены визуализации в {output_base}_*.svg/.png")

    logger.info("\n--- Лучшие и худшие примеры по метрикам ---")
    for metric, values in summary.items():
        valid = [(v, question_map[metric][i]) 
                for i, v in enumerate(values) if v is not None]
        if len(valid) >= 3:
            sorted_vals = sorted(valid, key=lambda x: x[0], reverse=True)
            logger.info(f"\n{metric}:")
            logger.info("  Лучшие 3:")
            for val, q in sorted_vals[:3]:
                logger.info(f"    {val:.4f} — {q[:80]}")
            logger.info("  Худшие 3:")
            for val, q in sorted_vals[-3:]:
                logger.info(f"    {val:.4f} — {q[:80]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate RAG model")
    parser.add_argument(
        '--dataset', type=str, required=True, 
        help='Path to test dataset JSON'
    )
    parser.add_argument(
        '--output', type=str, 
        default='evaluation/results/eval_results.json', 
        help='Path to save results'
    )
    parser.add_argument(
        '--exact_match', action='store_true', 
        help='Use exact string matching for context overlap (default is semantic)'
    )

    args = parser.parse_args()
    dataset = load_test_dataset(args.dataset)
    evaluate_all(dataset, args.output, args.exact_match)