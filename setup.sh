#!/bin/bash

# Проверка запуска Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "❌ Ollama не запущен. Запустите сервис Ollama перед продолжением."
  echo "   На macOS выполните: open -a Ollama"
  echo "   На Linux выполните: ollama serve"
  exit 1
fi

# Проверка наличия модели mistral:7b-instruct
if ! curl -s http://localhost:11434/api/tags | grep -q "mistral:7b-instruct"; then
  echo "⚠️ Модель mistral:7b-instruct не найдена в Ollama. Начинаем загрузку..."
  echo "📥 Загрузка модели mistral:7b-instruct может занять некоторое время (~4-5 ГБ)."
  curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral:7b-instruct"}'
  echo "✅ Модель mistral:7b-instruct успешно загружена."
fi

# Копирование .env файла, если он еще не существует
if [ ! -f .env ]; then
  echo "📄 Создание файла .env из примера..."
  cp .env.example .env
  echo "✅ Файл .env создан."
fi

# Проверка наличия директории db-init
if [ ! -d "db-init" ]; then
  echo "📂 Создаем директорию для скриптов инициализации базы данных..."
  mkdir -p db-init
  echo "✅ Директория db-init создана."
fi

# Проверка и обработка директории миграций
if [ -d "migrations" ]; then
  echo "🔄 Найдена директория миграций. В новой версии миграции встроены в приложение."
  echo "📝 Чтобы избежать конфликтов, директория миграций будет переименована."
  mv migrations migrations.backup.$(date +%Y%m%d%H%M%S)
  echo "✅ Миграции сохранены в резервной копии."
fi

# Запуск docker-compose
echo "🚀 Запуск сервисов..."
docker-compose up -d --build

echo "⏳ Ожидание запуска сервисов..."
sleep 10

echo "
✅ Система RAG запущена!

📚 Доступные сервисы:
   * Фронтенд: http://localhost:3000
   * API бэкенда: http://localhost:8000
   * Qdrant UI: http://localhost:6333/dashboard

🔍 Логи бэкенда: docker-compose logs -f backend
🖥️ Логи фронтенда: docker-compose logs -f frontend

❓ Демонстрационные учетные данные:
   * Логин: admin
   * Email: admin@example.com
   * Пароль: admin123

📋 Как начать:
   1. Откройте http://localhost:3000 в браузере
   2. Войдите в систему, используя демо-учетные данные
   3. Загрузите документы через интерфейс
   4. Задавайте вопросы по документам в чате
" 