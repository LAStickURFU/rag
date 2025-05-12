#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import time
import json
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluate.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("evaluator")

# Константы
DEFAULT_TEST_DATASET = "evaluation/test_dataset.json"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_RESULTS_FILE = "eval_results.json"
DEFAULT_UPLOAD_PROGRESS_FILE = "logs/upload_progress.json"
DEFAULT_USERNAME = "mass_upload"
DEFAULT_PASSWORD = "Qwerty123!"


def check_upload_completed(progress_file: str = DEFAULT_UPLOAD_PROGRESS_FILE) -> bool:
    """Проверяет, завершена ли загрузка файлов."""
    if not os.path.exists(progress_file):
        logger.warning(f"Файл прогресса {progress_file} не найден")
        return False
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            
        uploaded_files = progress.get("uploaded_files", [])
        errored_files = progress.get("errored_files", {})
        
        logger.info(f"Загружено файлов: {len(uploaded_files)}")
        logger.info(f"Файлов с ошибками: {len(errored_files)}")
        
        # Проверяем, что файлы были загружены
        return len(uploaded_files) > 0
    except Exception as e:
        logger.error(f"Ошибка при чтении файла прогресса: {e}")
        return False


def run_upload_script(text_dir: str, batch_size: int = 5, wait_time: int = 10, username: str = DEFAULT_USERNAME, password: str = DEFAULT_PASSWORD):
    """Запускает скрипт загрузки файлов, если он ещё не был запущен."""
    if not check_upload_completed():
        logger.info("Данные ещё не загружены, запускаем скрипт загрузки...")
        
        try:
            upload_script_path = os.path.join(os.path.dirname(__file__), "mass_upload.py")
            cmd = [
                sys.executable, 
                upload_script_path,
                "--text-dir", text_dir,
                "--batch-size", str(batch_size),
                "--wait-time", str(wait_time),
                "--logs-dir", os.path.dirname(DEFAULT_UPLOAD_PROGRESS_FILE),
                "--username", username,
                "--password", password
            ]
            
            logger.info(f"Запуск команды: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Ошибка при запуске скрипта загрузки: {result.stderr}")
                return False
            
            logger.info("Загрузка файлов завершена")
            return True
        except Exception as e:
            logger.error(f"Ошибка при запуске скрипта загрузки: {e}")
            return False
    else:
        logger.info("Данные уже загружены, пропускаем этап загрузки")
        return True


def run_evaluation(
    test_dataset: str = DEFAULT_TEST_DATASET,
    output_dir: str = DEFAULT_RESULTS_DIR,
    results_file: str = DEFAULT_RESULTS_FILE,
    exact_match: bool = False
):
    """Запускает оценку модели на основе тестового набора данных."""
    logger.info(f"Запуск оценки с использованием датасета: {test_dataset}")
    
    # Создаем директорию для результатов, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, results_file)
    
    # Формируем команду для запуска скрипта оценки
    evaluate_script_path = os.path.join("evaluation", "evaluate.py")
    cmd = [
        sys.executable,
        evaluate_script_path,
        "--dataset", test_dataset,
        "--output", output_path
    ]
    
    if exact_match:
        cmd.append("--exact_match")
    
    logger.info(f"Запуск команды: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Ошибка при запуске скрипта оценки: {result.stderr}")
            return False
        
        logger.info(f"Оценка завершена, результаты сохранены в {output_path}")
        
        # Выводим основные метрики из результатов
        if os.path.exists(output_path):
            summary_path = output_path.replace(".json", "_summary.csv")
            if os.path.exists(summary_path):
                logger.info(f"Сводка метрик сохранена в {summary_path}")
                
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1:
                        logger.info("Основные метрики:")
                        for line in lines[1:]:  # Пропускаем заголовок
                            parts = line.strip().split(',')
                            if len(parts) >= 2:
                                metric, mean = parts[0], parts[-1]
                                logger.info(f"  {metric}: {mean}")
                except Exception as e:
                    logger.error(f"Ошибка при чтении сводки метрик: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка при запуске скрипта оценки: {e}")
        return False


def create_test_dataset_if_not_exists(
    text_dir: str,
    output_file: str = DEFAULT_TEST_DATASET,
    num_questions: int = 50
):
    """
    Создает тестовый набор данных из случайных файлов, если он не существует.
    
    Args:
        text_dir: Директория с текстовыми файлами
        output_file: Путь для сохранения тестового набора
        num_questions: Количество вопросов в тестовом наборе
    """
    if os.path.exists(output_file):
        logger.info(f"Тестовый набор данных уже существует: {output_file}")
        return
    
    logger.info(f"Создание тестового набора данных из {text_dir}...")
    
    try:
        import random
        from tqdm import tqdm
        
        # Получаем список всех текстовых файлов
        text_files = list(Path(text_dir).glob("**/*.txt"))
        
        if not text_files:
            logger.error(f"Текстовые файлы не найдены в {text_dir}")
            return
        
        logger.info(f"Найдено {len(text_files)} текстовых файлов")
        
        # Выбираем случайные файлы для тестового набора
        selected_files = random.sample(text_files, min(num_questions, len(text_files)))
        
        test_data = []
        
        for file_path in tqdm(selected_files, desc="Создание тестовых вопросов"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    continue
                
                # Создаем простой вопрос по файлу
                # В реальном сценарии здесь можно использовать более сложную логику
                # для генерации вопросов, например с помощью LLM
                filename = file_path.stem
                
                # Генерируем простой вопрос на основе имени файла
                question = f"Что такое {filename.replace('-', ' ')}?"
                
                # Используем первые 100 символов как ожидаемый ответ
                expected_answer = content[:100] + "..." if len(content) > 100 else content
                
                test_data.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "source_file": str(file_path)
                })
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        
        # Создаем директорию для файла, если она не существует
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Сохраняем тестовый набор данных
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Создан тестовый набор данных с {len(test_data)} вопросами: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при создании тестового набора данных: {e}")


def main():
    parser = argparse.ArgumentParser(description="Загрузка данных и оценка RAG системы")
    parser.add_argument(
        "--text-dir",
        default="dataset_script/text",
        help="Директория с текстовыми файлами для загрузки"
    )
    parser.add_argument(
        "--test-dataset",
        default=DEFAULT_TEST_DATASET,
        help=f"Путь к тестовому набору данных (по умолчанию: {DEFAULT_TEST_DATASET})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Директория для сохранения результатов (по умолчанию: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help=f"Имя файла для сохранения результатов (по умолчанию: {DEFAULT_RESULTS_FILE})"
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Использовать точное совпадение строк для оценки контекста"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Количество вопросов в тестовом наборе данных"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Пропустить этап загрузки данных"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Размер пакета для загрузки файлов"
    )
    parser.add_argument(
        "--wait-time",
        type=int,
        default=10,
        help="Время ожидания между пакетами при загрузке (в секундах)"
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help=f"Имя пользователя для загрузки (по умолчанию: {DEFAULT_USERNAME})"
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help=f"Пароль пользователя для загрузки (по умолчанию: {DEFAULT_PASSWORD})"
    )
    
    args = parser.parse_args()
    
    # Создаем тестовый набор данных, если он не существует
    create_test_dataset_if_not_exists(
        args.text_dir,
        args.test_dataset,
        args.num_questions
    )
    
    # Запускаем загрузку данных, если нужно
    if not args.skip_upload:
        upload_success = run_upload_script(
            args.text_dir,
            args.batch_size,
            args.wait_time,
            args.username,
            args.password
        )
        
        if not upload_success:
            logger.error("Загрузка данных не удалась, оценка пропущена")
            return
    
    # Запускаем оценку
    run_evaluation(
        args.test_dataset,
        args.output_dir,
        args.results_file,
        args.exact_match
    )


if __name__ == "__main__":
    main() 