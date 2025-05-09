#!/usr/bin/env python3
"""
Скрипт для запуска всего приложения - бэкенд и фронтенд.

Использование:
    python -m scripts.run_app  # Запуск всего приложения
    python -m scripts.run_app --backend-only  # Только бэкенд
    python -m scripts.run_app --frontend-only  # Только фронтенд
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Глобальные переменные для процессов
backend_process = None
frontend_process = None
ollama_process = None


def signal_handler(sig, frame):
    """Обработчик сигналов для корректного завершения работы скрипта"""
    print("\nПолучен сигнал завершения. Останавливаем процессы...")
    stop_processes()
    sys.exit(0)


def start_ollama():
    """Запускает сервер Ollama, если он еще не запущен"""
    global ollama_process
    
    # Проверяем, запущен ли уже Ollama
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            print("✅ Ollama уже запущен")
            return True
    except Exception:
        pass
    
    print("Запускаем Ollama...")
    try:
        # На разных платформах запуск может отличаться
        if sys.platform == "darwin":  # macOS
            # Сначала пробуем через приложение
            try:
                subprocess.run(["open", "-a", "Ollama"], check=True)
                time.sleep(2)
                return True
            except Exception:
                pass
            
            # Если не получилось, пробуем через командную строку
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        elif sys.platform == "linux":
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        elif sys.platform == "win32":
            # На Windows может быть другой способ запуска
            ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Ждем запуска Ollama
        print("Ожидаем запуска Ollama...")
        max_retries = 10
        for i in range(max_retries):
            time.sleep(1)
            try:
                result = subprocess.run(
                    ["curl", "-s", "http://localhost:11434/api/tags"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if result.returncode == 0:
                    print("✅ Ollama успешно запущен")
                    return True
            except Exception:
                pass
            
            print(f"Ожидание Ollama... {i+1}/{max_retries}")
        
        print("❌ Не удалось запустить Ollama")
        return False
    except Exception as e:
        print(f"❌ Ошибка при запуске Ollama: {e}")
        return False


def start_backend():
    """Запускает бэкенд приложения"""
    global backend_process
    
    print("Запускаем бэкенд...")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    backend_log = open("logs/backend.log", "a")
    backend_process = subprocess.Popen(
        ["python", "-m", "app.main"],
        stdout=backend_log,
        stderr=backend_log
    )
    
    # Ждем запуска бэкенда
    print("Ожидаем запуска бэкенда...")
    max_retries = 10
    for i in range(max_retries):
        time.sleep(1)
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:8000/api/healthcheck"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                print("✅ Бэкенд успешно запущен")
                return True
        except Exception:
            pass
        
        print(f"Ожидание бэкенда... {i+1}/{max_retries}")
    
    print("❌ Не удалось запустить бэкенд")
    return False


def start_frontend():
    """Запускает фронтенд приложения"""
    global frontend_process
    
    print("Запускаем фронтенд...")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    frontend_log = open("logs/frontend.log", "a")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Директория фронтенда не найдена")
        return False
    
    frontend_process = subprocess.Popen(
        ["npm", "start"],
        cwd=str(frontend_dir),
        stdout=frontend_log,
        stderr=frontend_log,
        env={**os.environ, "BROWSER": "none"}  # Предотвращаем автоматическое открытие браузера
    )
    
    # Ждем запуска фронтенда
    print("Ожидаем запуска фронтенда...")
    max_retries = 20
    for i in range(max_retries):
        time.sleep(1)
        
        # Проверяем, что процесс все еще работает
        if frontend_process.poll() is not None:
            print("❌ Процесс фронтенда завершился с кодом:", frontend_process.returncode)
            return False
        
        # Проверяем, открыт ли порт 3000
        try:
            result = subprocess.run(
                ["lsof", "-i", ":3000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                print("✅ Фронтенд успешно запущен")
                print("📱 Откройте http://localhost:3000 в браузере")
                return True
        except Exception:
            pass
        
        print(f"Ожидание фронтенда... {i+1}/{max_retries}")
    
    print("❌ Не удалось запустить фронтенд")
    return False


def stop_processes():
    """Останавливает все запущенные процессы"""
    global backend_process, frontend_process, ollama_process
    
    if backend_process is not None:
        print("Останавливаем бэкенд...")
        try:
            backend_process.terminate()
            backend_process.wait(timeout=5)
        except Exception as e:
            print(f"Ошибка при остановке бэкенда: {e}")
            try:
                backend_process.kill()
            except Exception:
                pass
    
    if frontend_process is not None:
        print("Останавливаем фронтенд...")
        try:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)
        except Exception as e:
            print(f"Ошибка при остановке фронтенда: {e}")
            try:
                frontend_process.kill()
            except Exception:
                pass
    
    # Ollama обычно оставляем запущенной, но можно добавить флаг для ее остановки
    if ollama_process is not None:
        print("Останавливаем Ollama...")
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=5)
        except Exception as e:
            print(f"Ошибка при остановке Ollama: {e}")
            try:
                ollama_process.kill()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Run RAG application")
    parser.add_argument("--backend-only", action="store_true", help="Run only the backend")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend")
    parser.add_argument("--stop-ollama", action="store_true", help="Stop Ollama when exiting")
    args = parser.parse_args()
    
    # Регистрируем обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Запускаем компоненты в зависимости от аргументов
        if args.frontend_only:
            start_frontend()
        elif args.backend_only:
            start_ollama()
            start_backend()
        else:
            # Запускаем всё
            start_ollama()
            backend_started = start_backend()
            
            if backend_started:
                frontend_started = start_frontend()
                
                if not frontend_started:
                    print("Фронтенд не запустился, но бэкенд работает")
            else:
                print("Бэкенд не запустился, останавливаем всё")
                stop_processes()
                return 1
        
        # Ждем завершения работы
        print("\nПриложение запущено. Нажмите Ctrl+C для завершения.")
        
        # Главный цикл - просто ожидаем завершения процессов или сигнала
        while True:
            if backend_process is not None and backend_process.poll() is not None:
                print(f"⚠️ Бэкенд завершил работу с кодом {backend_process.returncode}")
                backend_process = None
            
            if frontend_process is not None and frontend_process.poll() is not None:
                print(f"⚠️ Фронтенд завершил работу с кодом {frontend_process.returncode}")
                frontend_process = None
            
            # Если все процессы завершились, выходим
            if (args.backend_only and backend_process is None) or \
               (args.frontend_only and frontend_process is None) or \
               (not args.backend_only and not args.frontend_only and 
                backend_process is None and frontend_process is None):
                print("Все процессы завершились, выходим")
                break
            
            time.sleep(1)
        
        return 0
    except KeyboardInterrupt:
        print("\nПолучено прерывание Ctrl+C")
    finally:
        stop_processes()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 