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
    filename=os.path.join(logs_dir, "evaluation.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/ask"
AUTH_URL = "http://localhost:8000/token"
USERNAME = "mass_upload"
PASSWORD = "Qwerty123!"
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

def get_auth_token():
    try:
        resp = requests.post(
            AUTH_URL,
            data={"username": USERNAME, "password": PASSWORD}
        )
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        logger.error(f"Failed to get auth token: {e}")
        return None

def ask_question(question: str):
    token = get_auth_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.post(API_URL, json={"question": question}, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        context = data.get("context", [])
        return {
            "answer": data.get("answer", "") or data.get("response", ""),
            "context": context
        }
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None

def ask_openai_judges(question, answer, context):
    """
    Использует GPT-3.5-turbo для оценки:
      - llm_relevance
      - llm_faithfulness
    """
    prompts = {
        "llm_relevance": (
            f"Оцени, насколько следующий ответ релевантен вопросу.\n\n"
            f"Вопрос: {question}\n"
            f"Ответ: {answer}\n\n"
            "Ответь числом от 0 до 1."
        ),
        "llm_faithfulness": (
            f"Оцени, насколько следующий ответ основан на контексте.\n\n"
            f"Контекст:\n"
            + "\n".join(f"{i+1}. {c}" for i,c in enumerate(context)) +
            f"\n\nОтвет: {answer}\n\n"
            "Ответь числом от 0 до 1."
        )
    }
    scores = {}
    for key, prompt in prompts.items():
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты — точный оценщик. Отвечай только числом от 0 до 1."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            val = float(resp.choices[0].message.content.strip())
            scores[key] = round(val, 4)
        except Exception as e:
            logger.error(f"Error in ask_openai_judges for {key}: {e}")
            scores[key] = None
    return scores

def load_test_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_metrics_heatmap(summary, output_base):
    data = {k: v for k, v in summary.items() if k != "mean" and v and len(v) > 0}
    if not data:
        logger.warning("No data for heatmap")
        return
    df = pd.DataFrame(data).replace({None: np.nan})
    plt.figure(figsize=(10, max(6, len(df) * 0.5)))
    sns.heatmap(df, annot=True, cmap="Blues", cbar=False)
    plt.tight_layout()
    plt.savefig(f"{output_base}_heatmap.svg")
    plt.savefig(f"{output_base}_heatmap.png")
    plt.close()

def plot_metrics_boxplot(summary, output_base):
    df = pd.DataFrame(summary).replace({None: np.nan})
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_base}_boxplot.svg")
    plt.savefig(f"{output_base}_boxplot.png")
    plt.close()

def plot_metrics_grouped(summary, output_base):
    """Создает группированные визуализации метрик по категориям."""
    data = {k: v for k, v in summary.items() if k != "mean" and v and len(v) > 0}
    if not data:
        logger.warning("Нет данных для группированной визуализации")
        return

    try:
        df = pd.DataFrame(data)

        # Группируем метрики по категориям, включая новые exact_match и answer_correctness
        metric_groups = {
            "Текстовое сходство": ["rouge_l", "bertscore", "exact_match", "answer_correctness"],
            "Контекст": ["context_recall_manual", "context_precision_manual"],
            "Семантика": ["faithfulness_semantic"],
            "LLM оценки": ["llm_relevance", "llm_faithfulness"],
            "RAGAS": ["ragas_answer_relevance", "ragas_faithfulness"]
        }

        for group_name, metrics_list in metric_groups.items():
            group_data = df[[m for m in metrics_list if m in df.columns]]
            if group_data.empty:
                continue

            plt.figure(figsize=(10, 6))
            if len(group_data) <= 5:
                # Точечный график для малого числа точек
                for col in group_data.columns:
                    plt.scatter(group_data.index, group_data[col], label=col)
                plt.legend()
            else:
                sns.boxplot(data=group_data)
            plt.title(f"Группа метрик: {group_name}")
            plt.ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            filename = f"{output_base}_group_{group_name.lower().replace(' ', '_')}"
            plt.savefig(f"{filename}.svg")
            plt.savefig(f"{filename}.png")
            plt.close()

        logger.info(f"Группированные визуализации сохранены в {output_base}_group_*.png/svg")
    except Exception as e:
        logger.error(f"Ошибка при создании группированных визуализаций: {e}")

def evaluate_one(item, exact_match):
    question = item['question']
    expected_answer = item['expected_answer']
    expected_context = item.get('expected_context', [])

    logger.info(f"Evaluating question: {question}")
    prediction = ask_question(question)
    if prediction is None:
        logger.warning(f"No prediction for question: {question}")
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

    # Сохраняем результаты
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Статистика по метрикам
    df = pd.DataFrame.from_dict(summary, orient='index')
    df["mean"] = df.apply(
        lambda row: round(
            sum(v for v in row if v is not None) / len([v for v in row if v is not None]), 4
        ) if any(v is not None for v in row) else None,
        axis=1
    )
    summary_csv = output_path.replace(".json", "_summary.csv")
    df.to_csv(summary_csv)
    logger.info(f"Summary CSV saved to {summary_csv}")

    # Визуализации
    base = output_path.rsplit(".", 1)[0]
    # Гистограмма средних значений
    try:
        plt.figure(figsize=(10, 6))
        df["mean"].sort_values().plot.barh()
        plt.title("Средние значения метрик")
        plt.xlabel("Значение")
        plt.tight_layout()
        plt.savefig(f"{base}_metrics.svg")
        plt.savefig(f"{base}_metrics.png")
        plt.close()
    except Exception as e:
        logger.error(f"Ошибка при построении графика средних метрик: {e}")

    plot_metrics_boxplot(summary, base)
    plot_metrics_grouped(summary, base)
    plot_metrics_heatmap(summary, base)

    # Лучшие и худшие примеры
    logger.info("=== Top-K and Bottom-K examples per metric ===")
    for metric, values in summary.items():
        valid = [(v, question_map[metric][i]) for i, v in enumerate(values) if v is not None]
        if len(valid) < 3:
            continue
        sorted_vals = sorted(valid, key=lambda x: x[0])
        logger.info(f"\nMetric: {metric}")
        logger.info("  Bottom 3:")
        for v, q in sorted_vals[:3]:
            logger.info(f"    {v:.4f} ← {q}")
        logger.info("  Top 3:")
        for v, q in sorted_vals[-3:]:
            logger.info(f"    {v:.4f} ← {q}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system outputs")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to test dataset JSON"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation/results/eval_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--exact_match", action="store_true",
        help="Use exact string matching for context overlap (default is semantic)"
    )
    args = parser.parse_args()

    ds = load_test_dataset(args.dataset)
    evaluate_all(ds, args.output, args.exact_match)
