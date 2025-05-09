import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gradio as gr
import numpy as np
from datetime import datetime


def load_evaluation_results(results_path="tests/rag_quality_results.json"):
    """Загружает результаты оценки из JSON файла"""
    if not os.path.exists(results_path):
        print(f"Файл с результатами не найден: {results_path}")
        return None
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    return results


def create_metrics_dataframe(results):
    """Преобразует результаты оценки в DataFrame для визуализации"""
    if not results:
        return None
    
    # Извлекаем скалярные метрики
    metrics = {}
    for key, value in results.items():
        if isinstance(value, (int, float)):
            metrics[key] = value
    
    # Создаем DataFrame
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])
    return df


def plot_radar_chart(df):
    """Создает радарный график метрик"""
    # Подготовка данных
    categories = df["Metric"].tolist()
    values = df["Score"].tolist()
    
    # Конвертируем в NumPy массивы
    categories = np.array(categories)
    values = np.array(values)
    
    # Количество категорий
    N = len(categories)
    
    # Угол для каждой категории
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Замыкаем график
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Строим график
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Устанавливаем метки
    ax.set_thetagrids(np.degrees(angles), categories)
    
    # Устанавливаем лимиты для осей
    ax.set_ylim(0, 1)
    
    # Добавляем заголовок
    ax.set_title("RAG Quality Metrics", size=15, y=1.1)
    
    plt.tight_layout()
    return fig


def plot_bar_chart(df):
    """Создает столбчатый график метрик"""
    plt.figure(figsize=(10, 6))
    
    # Используем seaborn для красивого графика
    ax = sns.barplot(x="Metric", y="Score", data=df, palette="viridis")
    
    # Добавляем значения над столбцами
    for i, v in enumerate(df["Score"]):
        ax.text(i, v + 0.05, f"{v:.3f}", ha="center")
    
    # Стилизуем график
    plt.title("RAG Quality Metrics", size=15)
    plt.ylim(0, 1.1)  # Устанавливаем предел оси Y для всех метрик
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()


def create_dashboard():
    """Создает интерфейс Gradio для визуализации метрик оценки RAG"""
    
    def update_visualizations():
        """Обновляет визуализации при загрузке новых данных"""
        results = load_evaluation_results()
        if not results:
            return "Ошибка загрузки результатов", None, None, None
        
        # Создаем DataFrame
        df = create_metrics_dataframe(results)
        if df is None:
            return "Ошибка создания DataFrame", None, None, None
        
        # Создаем таблицу с результатами
        table_html = df.to_html(classes="table table-striped", index=False)
        
        # Создаем графики
        radar_chart = plot_radar_chart(df)
        bar_chart = plot_bar_chart(df)
        
        threshold_analysis = ""
        for metric, score in zip(df["Metric"], df["Score"]):
            threshold = 0.7 if metric in ["faithfulness", "answer_relevancy", "answer_similarity"] else 0.5
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            threshold_analysis += f"{metric}: {score:.3f} - {status} (пороговое значение: {threshold})\n"
        
        return table_html, radar_chart, bar_chart, threshold_analysis
    
    # Создаем интерфейс
    with gr.Blocks(title="RAG Evaluation Dashboard") as dashboard:
        gr.Markdown("# Оценка качества RAG-системы")
        gr.Markdown("Метрики качества RAG на основе фреймворка RAGAS:")
        
        with gr.Row():
            with gr.Column():
                refresh_btn = gr.Button("Обновить данные")
                
                with gr.Tabs():
                    with gr.TabItem("Таблица метрик"):
                        metrics_table = gr.HTML()
                    
                    with gr.TabItem("Анализ пороговых значений"):
                        threshold_analysis = gr.Textbox(label="Анализ соответствия пороговым значениям", lines=10)
        
        with gr.Row():
            with gr.Column():
                radar_chart = gr.Plot(label="Радарный график метрик")
            
            with gr.Column():
                bar_chart = gr.Plot(label="Столбчатый график метрик")
        
        # Обработчик нажатия кнопки
        refresh_btn.click(
            fn=update_visualizations,
            outputs=[metrics_table, radar_chart, bar_chart, threshold_analysis]
        )
        
        # Автообновление при запуске
        dashboard.load(
            fn=update_visualizations,
            outputs=[metrics_table, radar_chart, bar_chart, threshold_analysis]
        )
    
    return dashboard


def main():
    """Основная функция для запуска дашборда"""
    dashboard = create_dashboard()
    dashboard.launch(share=False)


if __name__ == "__main__":
    main() 