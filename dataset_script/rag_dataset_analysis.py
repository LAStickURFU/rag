import os
import re
import json
import hashlib
from pathlib import Path
from collections import Counter
import pandas as pd
from langdetect import detect
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Папки с текстами и выводом
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Папка с текстами
TEXT_DIR = Path("text")

# Инициализация метрик
records = []
cumulative_vocab = Counter()
seen_hashes = set()


def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def get_doc_hash(text):
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def analyze_file(path: Path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    tokens = tokenize(cleaned)
    unique_words = set(tokens)
    doc_hash = get_doc_hash(cleaned)
    is_duplicate = doc_hash in seen_hashes

    if not is_duplicate:
        seen_hashes.add(doc_hash)

    # Подсчёт частот слов
    token_counts = Counter(tokens)
    cumulative_vocab.update(token_counts)

    # Анализ структуры
    num_lines = raw.count("\n")
    num_paragraphs = raw.count("\n\n")

    # Дополнительные метрики для ВКР
    ttr = len(unique_words) / len(tokens) if tokens else 0
    non_alpha_chars = len(re.findall(r"[^a-zA-Zа-яА-Я\s]", raw))
    noise_ratio = non_alpha_chars / len(raw) if raw else 0
    paragraph_word_ratio = num_paragraphs / len(tokens) if tokens else 0

    # Детекция языка
    try:
        lang = detect(raw)
    except:
        lang = "unknown"

    records.append({
        "filename": path.name,
        "num_chars": len(raw),
        "num_words": len(tokens),
        "num_unique_words": len(unique_words),
        "avg_word_length": sum(len(w) for w in tokens) / len(tokens) if tokens else 0,
        "top_5_words": dict(token_counts.most_common(5)),
        "num_lines": num_lines,
        "num_paragraphs": num_paragraphs,
        "language": lang,
        "is_duplicate": is_duplicate,
        "hash": doc_hash,
        "ttr": ttr,
        "noise_ratio": noise_ratio,
        "paragraph_word_ratio": paragraph_word_ratio,
    })


import time

start_time = time.time()

# Анализ всех файлов
text_files = list(TEXT_DIR.glob("*.txt"))
print(f"Найдено {len(text_files)} файлов. Начинаем анализ...")

for i, file in enumerate(text_files, 1):
    print(f"[{i}/{len(text_files)}] Обработка: {file.name}")
    analyze_file(file)

# Создание DataFrame
df = pd.DataFrame(records)

# Сохраняем CSV и JSON

# Подготовка отфильтрованного облака слов
custom_stopwords = set([
    "и", "в", "на", "по", "не", "что", "как", "это", "с", "из", "для", "у", "от", "до", "о", "за", "то",
    "а", "но", "же", "бы", "к", "быть", "или", "его", "её", "ее", "их", "так", "тоже", "там", "тут", "меня",
    "my", "this", "that", "the", "with", "you", "are", "have", "has", "was", "were", "from", "by", "of", "to",
    "a", "an", "is", "in", "at", "as", "it", "on", "your", "be", "can", "not", "we"
])

filtered_vocab = Counter({
    word: freq for word, freq in cumulative_vocab.items()
    if word not in custom_stopwords and len(word) > 2
})

wc_filtered = WordCloud(width=1000, height=600, background_color="white", max_words=200)
wc_filtered.generate_from_frequencies(filtered_vocab)

# Сохраняем отфильтрованное облако
wc_filtered.to_file(str(OUTPUT_DIR / "wordcloud_filtered.png"))
plt.figure(figsize=(10, 6))
plt.imshow(wc_filtered, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wordcloud_filtered.svg")
plt.savefig(OUTPUT_DIR / "wordcloud_filtered.pdf")
df.to_csv(OUTPUT_DIR / "dataset_stats.csv", index=False)
df.to_json(OUTPUT_DIR / "dataset_stats.json", indent=2, force_ascii=False)

wc = WordCloud(width=1000, height=600, background_color="white", max_words=200)
wc.generate_from_frequencies(cumulative_vocab)

# Сохраняем как изображение (PNG)
wc.to_file(str(OUTPUT_DIR / "wordcloud.png"))

# Альтернативно: сохраняем WordCloud через matplotlib в SVG и PDF
plt.figure(figsize=(10, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wordcloud.svg")
plt.savefig(OUTPUT_DIR / "wordcloud.pdf")

# Дополнительные RAG-метрики

# Вариации длин чанков
chunk_buckets = {
    "<100": (0, 100),
    "100-299": (100, 300),
    "300-499": (300, 500),
    "500-699": (500, 700),
    "700-999": (700, 1000),
    "1000-1499": (1000, 1500),
    "1500-1999": (1500, 2000),
    ">2000": (2000, float("inf"))
}

chunk_length_distribution = {
    label: df[(df["num_words"] >= bounds[0]) & (df["num_words"] < bounds[1])].shape[0]
    for label, bounds in chunk_buckets.items()
}
chunk_size = 512
chunk_counts = df["num_words"].apply(lambda x: (x + chunk_size - 1) // chunk_size)

# Статистика по разным размерам чанков
chunk_sizes = [256, 512, 1024]
chunk_stats_by_size = {}
for size in chunk_sizes:
    counts = df["num_words"].apply(lambda x: (x + size - 1) // size)
    chunk_stats_by_size[size] = {
        "avg_chunks_per_doc": counts.mean(),
        "max_chunks": counts.max(),
        "min_chunks": counts.min(),
        "num_docs_with_1_chunk": (counts == 1).sum(),
        "num_docs_with_2_or_more": (counts >= 2).sum()
    }
rag_stats = {
    "avg_chunks_per_doc": chunk_counts.mean(),
    "max_chunks": chunk_counts.max(),
    "min_chunks": chunk_counts.min(),
    "num_docs_over_1_chunk": (chunk_counts > 1).sum(),
    "num_docs_under_100_words": (df["num_words"] < 100).sum(),
    "duplicate_ratio": df["is_duplicate"].sum() / len(df),
    "avg_paragraphs_per_doc": df["num_paragraphs"].mean(),
    "chunk_length_distribution": chunk_length_distribution,
    "chunk_stats_by_size": chunk_stats_by_size
}

# Визуализация

# Гистограммы дополнительных метрик: TTR, шумность, структурность
plt.figure(figsize=(10, 6))
df["ttr"].hist(bins=50, color="steelblue")
plt.title("Коэффициент лексического разнообразия (TTR)")
plt.xlabel("TTR")
plt.ylabel("Число документов")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ttr_distribution.png")
plt.savefig(OUTPUT_DIR / "ttr_distribution.svg")
plt.savefig(OUTPUT_DIR / "ttr_distribution.pdf")

plt.figure(figsize=(10, 6))
df["noise_ratio"].hist(bins=50, color="darkorange")
plt.title("Доля неалфавитных символов (шумность)")
plt.xlabel("Шумность")
plt.ylabel("Число документов")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "noise_ratio_distribution.png")
plt.savefig(OUTPUT_DIR / "noise_ratio_distribution.svg")
plt.savefig(OUTPUT_DIR / "noise_ratio_distribution.pdf")

plt.figure(figsize=(10, 6))
df["paragraph_word_ratio"].hist(bins=50, color="seagreen")
plt.title("Отношение абзацев к количеству слов")
plt.xlabel("Абзацы / Слова")
plt.ylabel("Число документов")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "paragraph_word_ratio_distribution.png")
plt.savefig(OUTPUT_DIR / "paragraph_word_ratio_distribution.svg")
plt.savefig(OUTPUT_DIR / "paragraph_word_ratio_distribution.pdf")

# Сравнительный график распределения чанков при разных размерах
plt.figure(figsize=(10, 6))
for size in chunk_sizes:
    counts = df["num_words"].apply(lambda x: (x + size - 1) // size)
    distribution = counts.value_counts().sort_index()
    plt.plot(distribution.index, distribution.values, label=f"chunk size = {size}")

plt.title("Сравнение распределения количества чанков на документ")
plt.xlabel("Количество чанков")
plt.ylabel("Число документов")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chunk_comparison_line.png")
plt.savefig(OUTPUT_DIR / "chunk_comparison_line.svg")
plt.savefig(OUTPUT_DIR / "chunk_comparison_line.pdf")

# Гистограмма длины документов (в словах)
plt.figure(figsize=(10, 6))
df["num_words"].hist(bins=50)
plt.title("Распределение длины документов (в словах)")
plt.xlabel("Число слов")
plt.ylabel("Число документов")
plt.savefig(OUTPUT_DIR / "hist_num_words.png")
plt.savefig(OUTPUT_DIR / "hist_num_words.svg")
plt.savefig(OUTPUT_DIR / "hist_num_words.pdf")

# График распределения чанков по длине
plt.figure(figsize=(10, 6))
labels = list(chunk_length_distribution.keys())
values = list(chunk_length_distribution.values())
plt.bar(labels, values, color="skyblue")
plt.title("Распределение документов по длине (в словах)")
plt.xlabel("Диапазон длины (слов)")
plt.ylabel("Количество документов")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chunk_length_distribution.png")
plt.savefig(OUTPUT_DIR / "chunk_length_distribution.svg")
plt.savefig(OUTPUT_DIR / "chunk_length_distribution.pdf")

# Визуализация распределения количества чанков на документ
plt.figure(figsize=(10, 6))
chunk_counts.value_counts().sort_index().plot(kind="bar", color="salmon")
plt.title("Распределение количества чанков на документ (chunk size = 512)")
plt.xlabel("Количество чанков")
plt.ylabel("Число документов")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_chunks_per_doc.png")
plt.savefig(OUTPUT_DIR / "hist_chunks_per_doc.svg")
plt.savefig(OUTPUT_DIR / "hist_chunks_per_doc.pdf")

# Сводка
summary = {
    "total_docs": len(df),
    "avg_words": df["num_words"].mean(),
    "max_words": df["num_words"].max(),
    "min_words": df["num_words"].min(),
    "avg_chars": df["num_chars"].mean(),
    "num_duplicates": df["is_duplicate"].sum(),
    "languages": df["language"].value_counts().to_dict(),
    "total_vocab_size": len(cumulative_vocab),
    "top_words": dict(cumulative_vocab.most_common(20)),
    "avg_ttr": df["ttr"].mean(),
    "min_ttr": df["ttr"].min(),
    "max_ttr": df["ttr"].max(),
    "avg_noise_ratio": df["noise_ratio"].mean(),
    "min_noise_ratio": df["noise_ratio"].min(),
    "max_noise_ratio": df["noise_ratio"].max(),
    "avg_paragraph_word_ratio": df["paragraph_word_ratio"].mean(),
    "min_paragraph_word_ratio": df["paragraph_word_ratio"].min(),
    "max_paragraph_word_ratio": df["paragraph_word_ratio"].max()
}


# Преобразование типов в стандартные Python-типы

def convert(obj):
    if isinstance(obj, (pd.Series, dict)):
        return {k: convert(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        return obj.item()
    return obj


summary.update({"rag_metrics": convert(rag_stats)})
summary = convert(summary)

with open(OUTPUT_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start_time
print(f"Анализ завершён за {elapsed:.2f} секунд. Сводка и графики сохранены.")
