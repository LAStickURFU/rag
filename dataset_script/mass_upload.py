#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import argparse
import logging
import sys
import random
from pathlib import Path
from typing import List, Dict
import re

import requests
from tqdm import tqdm

# Определяем путь к директории логов
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Создаем директорию для логов, если она не существует
if not LOGS_DIR.exists():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "mass_uploader_log.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mass_uploader")

# Константы
API_BASE_URL = "http://localhost:8000"
AUTH_ENDPOINT = f"{API_BASE_URL}/token"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/documents"
UPLOAD_ENDPOINT = f"{DOCUMENTS_ENDPOINT}/upload"
LIST_DOCUMENTS_ENDPOINT = DOCUMENTS_ENDPOINT

DEFAULT_USERNAME = "mass_upload"
DEFAULT_PASSWORD = "Qwerty123!"
REGISTER_ENDPOINT = f"{API_BASE_URL}/register"

# Параметры пакетной обработки
DEFAULT_BATCH_SIZE = 60  # Количество файлов в одной пакетной загрузке
DEFAULT_WAIT_BETWEEN_BATCHES = 2  # Секунды ожидания между пакетами
DEFAULT_RETRY_COUNT = 3  # Количество повторных попыток при ошибках
DEFAULT_TEXT_DIR = "dataset_script/text"
DEFAULT_RESUME_FILE = LOGS_DIR / "upload_progress.json"


class MassUploader:
    """Класс для массовой загрузки текстовых файлов в RAG систему."""

    def __init__(
        self,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        text_dir: str = DEFAULT_TEXT_DIR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        wait_time: int = DEFAULT_WAIT_BETWEEN_BATCHES,
        retries: int = DEFAULT_RETRY_COUNT,
        resume_file: str = DEFAULT_RESUME_FILE,
    ):
        self.username = username
        self.password = password
        self.text_dir = Path(text_dir)
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.retries = retries
        
        # Преобразуем путь к файлу прогресса в Path, если он передан как строка
        if isinstance(resume_file, str):
            self.resume_file = Path(resume_file)
        else:
            self.resume_file = resume_file
        
        self.access_token = None
        self.uploaded_files = set()
        self.errored_files = {}
        
        # Загружаем прогресс, если файл существует
        self._load_progress()
        
    def _load_progress(self):
        """Загружает прогресс из файла, если он существует."""
        if self.resume_file.exists():
            try:
                with open(self.resume_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    self.uploaded_files = set(progress.get("uploaded_files", []))
                    self.errored_files = progress.get("errored_files", {})
                    logger.info(
                        f"Загружен прогресс: {len(self.uploaded_files)} "
                        f"загруженных файлов, "
                        f"{len(self.errored_files)} файлов с ошибками"
                    )
            except Exception as e:
                logger.error(f"Ошибка при загрузке прогресса: {e}")
    
    def _save_progress(self):
        """Сохраняет текущий прогресс в файл."""
        try:
            with open(self.resume_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "uploaded_files": list(self.uploaded_files),
                        "errored_files": self.errored_files,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"Прогресс сохранен в {self.resume_file}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении прогресса: {e}")
    
    def _ensure_user_exists(self):
        """Проверяет существование пользователя и создает его при необходимости."""
        try:
            # Пробуем авторизоваться с существующими учетными данными
            token_response = requests.post(
                AUTH_ENDPOINT,
                data={"username": self.username, "password": self.password},
            )
            
            if token_response.status_code == 200:
                logger.info(
                    f"Пользователь {self.username} существует, "
                    f"авторизация успешна"
                )
                self.access_token = token_response.json().get("access_token")
                return True
            
            # Если пользователь не существует, создаем его
            logger.warning(
                f"Пользователь {self.username} не найден, "
                f"создаем нового пользователя"
            )
            register_response = requests.post(
                REGISTER_ENDPOINT,
                json={"username": self.username, "password": self.password},
            )
            
            if register_response.status_code == 200:
                logger.info(f"Пользователь {self.username} успешно создан")
                # Авторизуемся с новым пользователем
                token_response = requests.post(
                    AUTH_ENDPOINT,
                    data={"username": self.username, "password": self.password},
                )
                
                if token_response.status_code == 200:
                    self.access_token = token_response.json().get("access_token")
                    logger.info("Авторизация с новым пользователем успешна")
                    return True
                else:
                    logger.error(
                        f"Ошибка авторизации после создания пользователя: "
                        f"{token_response.status_code} - {token_response.text}"
                    )
            else:
                logger.error(
                    f"Ошибка создания пользователя: "
                    f"{register_response.status_code} - {register_response.text}"
                )
            
            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке/создании пользователя: {e}")
            return False
    
    def _get_all_text_files(self) -> List[Path]:
        """Получает список всех текстовых файлов из указанной директории."""
        try:
            if not self.text_dir.exists():
                logger.error(f"Директория {self.text_dir} не существует")
                return []
            
            all_files = list(self.text_dir.glob("**/*.txt"))
            logger.info(f"Всего найдено {len(all_files)} текстовых файлов")
            return all_files
        except Exception as e:
            logger.error(f"Ошибка при поиске файлов: {e}")
            return []
    
    def _get_files_to_upload(self) -> List[Path]:
        """Возвращает список файлов, которые нужно загрузить."""
        all_files = self._get_all_text_files()
        files_to_upload = [
            f for f in all_files if f.name not in self.uploaded_files
        ]
        logger.info(
            f"Осталось загрузить {len(files_to_upload)} из {len(all_files)} файлов"
        )
        return files_to_upload
    
    def _check_documents_status(self) -> Dict[str, int]:
        """Проверяет статус документов на сервере."""
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            # Добавляем параметр return_all=true для получения всех документов без пагинации
            response = requests.get(f"{LIST_DOCUMENTS_ENDPOINT}?return_all=true", headers=headers)
            
            if response.status_code != 200:
                logger.error(
                    f"Ошибка при получении списка документов: "
                    f"{response.status_code} - {response.text}"
                )
                return {}
            
            documents = response.json()
            status_counts = {}
            
            for doc in documents:
                status = doc.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return status_counts
        except Exception as e:
            logger.error(f"Ошибка при проверке статуса документов: {e}")
            return {}
    
    def _wait_for_processing(self, max_wait: int = 300, check_interval: int = 10):
        """
        Ожидает завершения обработки всех документов.
        
        Args:
            max_wait: Максимальное время ожидания в секундах
            check_interval: Интервал проверки статуса в секундах
        """
        start_time = time.time()
        logger.info("Ожидание завершения обработки документов...")
        
        while True:
            status_counts = self._check_documents_status()
            
            # Проверяем, есть ли документы в процессе обработки
            processing_docs = sum(
                status_counts.get(status, 0) 
                for status in [
                    "uploaded", "processing", "chunking", 
                    "embedding", "reindexing"
                ]
            )
            
            logger.info(f"Статусы документов: {status_counts}")
            
            if processing_docs == 0:
                logger.info("Все документы обработаны")
                break
            
            # Проверяем таймаут
            if time.time() - start_time > max_wait:
                logger.warning(
                    f"Превышено максимальное время ожидания ({max_wait} сек). "
                    f"Продолжение с необработанными документами."
                )
                break
            
            logger.info(f"Ожидание... {processing_docs} документов в обработке")
            time.sleep(check_interval)
    
    def _upload_batch(self, batch: List[Path]) -> bool:
        """
        Загружает пакет файлов.
        
        Args:
            batch: Список путей к файлам для загрузки
            
        Returns:
            bool: True, если все файлы в пакете загружены успешно
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        files = []
        titles = []
        
        try:
            # Подготовка файлов и заголовков
            for file_path in batch:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    title = file_path.stem  # Имя файла без расширения
                    
                    files.append(
                        ("files", (file_path.name, content, "text/plain"))
                    )
                    titles.append(title)
                except Exception as e:
                    logger.error(f"Ошибка при подготовке файла {file_path}: {e}")
                    self.errored_files[file_path.name] = str(e)
                    return False
            
            # Добавляем заголовки к запросу
            form_data = []
            for title in titles:
                form_data.append(("titles", title))
            
            # Отправляем запрос на загрузку
            response = requests.post(
                UPLOAD_ENDPOINT, 
                headers=headers,
                files=files,
                data=form_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Проверяем результат для каждого файла
                uploaded_docs = result.get("documents", [])
                for doc in uploaded_docs:
                    if doc.get("status") == "error":
                        filename = doc.get("filename", "unknown")
                        error = doc.get("error", "unknown error")
                        logger.error(f"Ошибка при загрузке файла {filename}: {error}")
                        self.errored_files[filename] = error
                    else:
                        filename = doc.get("filename")
                        if filename:
                            logger.info(f"Файл {filename} успешно загружен")
                            self.uploaded_files.add(filename)
                
                return True
            else:
                logger.error(
                    f"Ошибка при загрузке пакета: "
                    f"{response.status_code} - {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Общая ошибка при загрузке пакета: {e}")
            return False
    
    def upload_all_files(self):
        """Загружает все текстовые файлы из указанной директории."""
        if not self._ensure_user_exists():
            logger.error("Не удалось авторизоваться, загрузка прервана")
            return
        
        files_to_upload = self._get_files_to_upload()
        
        if not files_to_upload:
            logger.info("Нет файлов для загрузки")
            return
        
        # Добавляем яркое сообщение о режиме работы скрипта
        resume_mode = len(self.uploaded_files) > 0
        resume_msg = "ПРОДОЛЖЕНИЕ ЗАГРУЗКИ" if resume_mode else "НОВАЯ ЗАГРУЗКА С НУЛЯ"
        
        # Добавляем заголовок с отчетливой разделительной линией
        logger.info("")
        logger.info("="*50)
        logger.info(f"     {resume_msg}")
        logger.info("-"*50)
        
        # Выводим полезную статистику
        all_files = self._get_all_text_files()
        logger.info(f"     ВСЕГО ФАЙЛОВ:      {len(all_files)}")
        logger.info(f"     УЖЕ ЗАГРУЖЕНО:     {len(self.uploaded_files)}")
        logger.info(f"     ОСТАЛОСЬ ЗАГРУЗИТЬ: {len(files_to_upload)}")
        
        # Если есть ошибки из предыдущих запусков, сообщаем о них
        if self.errored_files:
            logger.info(f"     ОШИБКИ ЗАГРУЗКИ:    {len(self.errored_files)}")
            # Группируем ошибки по типам для более компактного вывода
            error_types = {}
            for file_name, error in self.errored_files.items():
                error_types[error] = error_types.get(error, 0) + 1
            
            # Выводим топ-3 типа ошибок
            if error_types:
                logger.info(f"     ТОП ОШИБОК:")
                for i, (error, count) in enumerate(
                    sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
                ):
                    short_error = error[:50] + "..." if len(error) > 50 else error
                    logger.info(f"     {i+1}. {short_error} - {count} файл(ов)")
        
        logger.info("="*50)
        logger.info("")
        
        # Перемешиваем файлы для более равномерного распределения размеров
        random.shuffle(files_to_upload)
        
        logger.info(f"Начинаем загрузку {len(files_to_upload)} файлов...")
        
        # Загружаем файлы небольшими пакетами
        batches = [
            files_to_upload[i:i+self.batch_size] 
            for i in range(0, len(files_to_upload), self.batch_size)
        ]
        batch_pbar = tqdm(batches, desc="Загрузка пакетов", unit="пакет")
        
        for batch_idx, batch in enumerate(batch_pbar):
            batch_pbar.set_postfix({"пакет": f"{batch_idx+1}/{len(batches)}"})
            
            # Показываем детали текущего пакета
            file_names = [f.name for f in batch]
            logger.info(f"Загрузка пакета {batch_idx+1}: {file_names}")
            
            retry_count = 0
            success = False
            
            while retry_count < self.retries and not success:
                if retry_count > 0:
                    logger.info(
                        f"Повторная попытка {retry_count}/{self.retries} "
                        f"для пакета {batch_idx+1}"
                    )
                
                success = self._upload_batch(batch)
                
                if not success:
                    retry_count += 1
                    if retry_count < self.retries:
                        time.sleep(2)  # Небольшая пауза перед повторной попыткой
            
            # Сохраняем прогресс после каждого пакета
            self._save_progress()
            
            # Ждем между пакетами для стабильности
            if batch_idx < len(batches) - 1:  # Если это не последний пакет
                logger.info(f"Ожидание {self.wait_time} секунд перед следующим пакетом...")
                
                # Каждые 5 пакетов проверяем статус документов
                if (batch_idx + 1) % 5 == 0:
                    self._wait_for_processing()
                else:
                    time.sleep(self.wait_time)
        
        # Финальное ожидание обработки всех документов
        logger.info("Загрузка завершена, ожидание завершения обработки всех документов...")
        self._wait_for_processing(max_wait=600)  # 10 минут максимум
        
        # Итоговая статистика
        logger.info("=== Итоговая статистика ===")
        logger.info(f"Всего найдено файлов: {len(self._get_all_text_files())}")
        logger.info(f"Успешно загружено: {len(self.uploaded_files)}")
        logger.info(f"Файлов с ошибками: {len(self.errored_files)}")
        
        if self.errored_files:
            logger.info("Файлы с ошибками:")
            for filename, error in self.errored_files.items():
                logger.info(f" - {filename}: {error}")
        
        # Проверка статусов на сервере
        status_counts = self._check_documents_status()
        logger.info(f"Статусы документов на сервере: {status_counts}")


def analyze_upload_log(log_file=None):
    """
    Анализирует лог загрузки документов и выводит статистику.
    
    Args:
        log_file: Путь к файлу лога. Если None, используется стандартный путь.
    """
    if log_file is None:
        log_file = LOGS_DIR / "mass_uploader_log.log"
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Файл лога {log_path} не найден")
        return
    
    # Счетчики для анализа
    uploaded_count = 0
    error_count = 0
    errors = {}
    file_stats = {}
    processing_times = {}
    
    # Паттерны для анализа логов
    upload_pattern = re.compile(r"Файл (.*?) успешно загружен")
    error_pattern = re.compile(r"Ошибка при загрузке файла (.*?): (.*)")
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # Поиск успешных загрузок
                upload_match = upload_pattern.search(line)
                if upload_match:
                    filename = upload_match.group(1)
                    uploaded_count += 1
                    file_stats[filename] = {"status": "успешно"}
                
                # Поиск ошибок
                error_match = error_pattern.search(line)
                if error_match:
                    filename = error_match.group(1)
                    error_msg = error_match.group(2)
                    error_count += 1
                    errors[filename] = error_msg
                    file_stats[filename] = {"status": "ошибка", "error": error_msg}
    
        # Вывод статистики
        print(f"\n=== Анализ лога загрузки ===")
        print(f"Файл лога: {log_path}")
        print(f"Всего успешно загружено файлов: {uploaded_count}")
        print(f"Всего файлов с ошибками: {error_count}")
        
        if error_count > 0:
            print("\nТип-5 частых ошибок:")
            error_types = {}
            for err_msg in errors.values():
                error_types[err_msg] = error_types.get(err_msg, 0) + 1
            
            for i, (err_msg, count) in enumerate(
                sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
            ):
                print(f"{i+1}. {err_msg}: {count} файлов")
        
        return {
            "uploaded_count": uploaded_count,
            "error_count": error_count,
            "errors": errors,
            "file_stats": file_stats
        }
    
    except Exception as e:
        print(f"Ошибка при анализе лога: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Массовая загрузка текстовых файлов в RAG систему")
    parser.add_argument(
        "--text-dir", 
        default=DEFAULT_TEXT_DIR,
        help=f"Директория с текстовыми файлами (по умолчанию: {DEFAULT_TEXT_DIR})"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help=f"Размер пакета для загрузки (по умолчанию: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--wait-time", 
        type=int, 
        default=DEFAULT_WAIT_BETWEEN_BATCHES,
        help=f"Время ожидания между пакетами в секундах (по умолчанию: {DEFAULT_WAIT_BETWEEN_BATCHES})"
    )
    parser.add_argument(
        "--username", 
        default=DEFAULT_USERNAME,
        help=f"Имя пользователя для загрузки (по умолчанию: {DEFAULT_USERNAME})"
    )
    parser.add_argument(
        "--password", 
        default=DEFAULT_PASSWORD,
        help=f"Пароль пользователя (по умолчанию: {DEFAULT_PASSWORD})"
    )
    parser.add_argument(
        "--resume-file", 
        default=DEFAULT_RESUME_FILE,
        help=f"Файл для сохранения прогресса (по умолчанию: {DEFAULT_RESUME_FILE})"
    )
    parser.add_argument(
        "--retries", 
        type=int, 
        default=DEFAULT_RETRY_COUNT,
        help=f"Количество повторных попыток при ошибках (по умолчанию: {DEFAULT_RETRY_COUNT})"
    )
    parser.add_argument(
        "--analyze-log",
        action="store_true",
        help="Проанализировать лог загрузки и вывести статистику"
    )
    parser.add_argument(
        "--log-file",
        help="Путь к файлу лога для анализа (по умолчанию используется стандартный путь)"
    )
    parser.add_argument(
        "--logs-dir",
        default=str(LOGS_DIR),
        help=f"Директория для хранения логов (по умолчанию: {LOGS_DIR})"
    )
    
    args = parser.parse_args()
    
    # Если запрошен анализ логов
    if args.analyze_log:
        analyze_upload_log(args.log_file)
        sys.exit(0)
    
    # Настраиваем путь к файлу прогресса в директории логов
    if args.logs_dir != str(LOGS_DIR):
        custom_logs_dir = Path(args.logs_dir)
        if not custom_logs_dir.exists():
            custom_logs_dir.mkdir(parents=True, exist_ok=True)
            
        resume_file = custom_logs_dir / "upload_progress.json"
        log_handler = logging.FileHandler(custom_logs_dir / "mass_uploader_log.log")
        logger.handlers = [log_handler, logging.StreamHandler(sys.stdout)]
        logger.info(f"Используется директория логов: {custom_logs_dir}")
    else:
        resume_file = DEFAULT_RESUME_FILE
    
    uploader = MassUploader(
        username=args.username,
        password=args.password,
        text_dir=args.text_dir,
        batch_size=args.batch_size,
        wait_time=args.wait_time,
        retries=args.retries,
        resume_file=resume_file,
    )
    
    try:
        uploader.upload_all_files()
    except KeyboardInterrupt:
        logger.info("Загрузка прервана пользователем")
        uploader._save_progress()
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        uploader._save_progress() 