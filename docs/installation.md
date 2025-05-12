# Руководство по установке

В этом документе описаны шаги для установки и запуска RAG-системы.

## Требования

Для работы системы необходимы:

- Docker и Docker Compose (версия 3.8+)
- Python 3.9+ (для запуска без Docker)
- Ollama - для запуска локальных языковых моделей
- 8+ ГБ свободной оперативной памяти
- 10+ ГБ свободного дискового пространства (без учета моделей Ollama)

## Быстрая установка (рекомендуется)

Система включает скрипт автоматической настройки и запуска, который упрощает процесс развертывания.

### Шаг 1: Установка Ollama

Ollama требуется для запуска локальных языковых моделей.

#### macOS

1. Скачайте Ollama с [официального сайта](https://ollama.ai/download)
2. Запустите установщик и следуйте инструкциям
3. Запустите Ollama через Launchpad или командой:
   ```bash
   open -a Ollama
   ```

#### Linux

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

#### Windows (через WSL2)

1. Установите WSL2
2. В терминале WSL выполните:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   ```

### Шаг 2: Клонирование репозитория

```bash
git clone https://your-repository-url.git rag
cd rag
```

### Шаг 3: Запуск скрипта установки

```bash
./setup.sh
```

Скрипт автоматически:
- Проверит наличие и запуск Ollama
- Загрузит необходимую модель (mistral:7b-instruct)
- Создаст конфигурационный файл `.env`
- Запустит все контейнеры через Docker Compose

### Шаг 4: Доступ к системе

После успешного запуска, система будет доступна по следующим адресам:

- **Фронтенд**: [http://localhost:3000](http://localhost:3000)
- **API бэкенда**: [http://localhost:8000](http://localhost:8000)
- **Qdrant UI**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Ручная установка

Если вы предпочитаете контролировать каждый шаг установки, следуйте инструкциям ниже.

### Шаг 1: Настройка переменных окружения

Создайте файл `.env` на основе примера:

```bash
cp .env.example .env
```

Отредактируйте его, установив необходимые значения:

```ini
# Настройки базы данных
DATABASE_URL=postgresql+psycopg2://postgres:mysecretpassword@postgres:5432/postgres

# Настройки Ollama
OLLAMA_MODEL=mistral:7b-instruct
OLLAMA_HOST=http://host.docker.internal:11434

# Настройки RAG
EMBEDDING_MODEL=intfloat/multilingual-e5-large
CHUNK_SIZE=400
CHUNK_OVERLAP=100
USE_HYBRID_SEARCH=true
USE_RERANKER=true
CHUNKING_MODE=character

# Другие настройки
LOG_LEVEL=INFO
LOG_TO_FILE=true
```

### Шаг 2: Запуск Ollama

Убедитесь, что Ollama запущен и загрузите необходимую модель:

```bash
# Запуск Ollama (если не запущен)
ollama serve

# Загрузка модели
curl -X POST http://localhost:11434/api/pull -d '{"name": "mistral:7b-instruct"}'
```

### Шаг 3: Запуск системы через Docker Compose

```bash
docker-compose up -d
```

## Настройка после установки

### Демонстрационный аккаунт

Система создает демонстрационный аккаунт администратора при первом запуске:

- **Логин**: admin
- **Email**: admin@example.com
- **Пароль**: admin123

Рекомендуется изменить пароль после первого входа.

### Проверка статуса системы

Чтобы убедиться, что все компоненты работают правильно:

```bash
# Проверка статуса контейнеров
docker-compose ps

# Просмотр логов бэкенда
docker-compose logs -f backend

# Проверка API
curl http://localhost:8000/api/healthcheck
```

## Запуск без Docker

Для разработки или отладки, система может быть запущена без Docker:

### Шаг 1: Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

### Шаг 2: Настройка переменных окружения

```bash
cp .env.example .env
```

Внесите необходимые изменения для локального запуска:
- Измените `DATABASE_URL` на локальный PostgreSQL
- Измените `QDRANT_HOST` на `localhost`

### Шаг 3: Запуск компонентов

Вам понадобится локально установленные PostgreSQL и Qdrant, а также Ollama.

```bash
# Запуск приложения
python -m app.main
```

## Устранение неполадок

### Проблема: Ошибка подключения к Ollama

**Симптомы**: Сообщение "Failed to connect to Ollama" в логах или интерфейсе.

**Решение**:
1. Убедитесь, что Ollama запущен (`ollama serve`)
2. Проверьте адрес в переменной окружения `OLLAMA_HOST`
3. Для Docker на macOS/Windows убедитесь, что используется `host.docker.internal` вместо `localhost`

### Проблема: Ошибки с зависимостями Python

**Симптомы**: Ошибки импорта или конфликты версий

**Решение**:
```bash
pip install huggingface_hub==0.14.1 transformers==4.30.2 sentence-transformers==2.2.2 tokenizers==0.13.3
```

### Проблема: Контейнеры не запускаются

**Решение**:
```bash
# Остановка и удаление всех контейнеров
docker-compose down

# Удаление томов (осторожно - все данные будут потеряны!)
docker-compose down -v

# Перезапуск
docker-compose up -d
```

## Обновление системы

### Обновление через Git

```bash
git pull
docker-compose build
docker-compose up -d
```

### Обновление моделей Ollama

```bash
ollama pull mistral:7b-instruct
```

## Дополнительные опции

### Запуск с GPU поддержкой

Для использования GPU, добавьте в `docker-compose.yml` для контейнера `backend`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

И укажите в `.env`:
```
USE_GPU=true
```

### Изменение порта фронтенда

Если порт 3000 занят, измените его в `docker-compose.yml`:

```yaml
frontend:
  ports:
    - "8080:80"  # Замените 3000 на 8080
```

После изменения перезапустите контейнеры:
```bash
docker-compose up -d
``` 