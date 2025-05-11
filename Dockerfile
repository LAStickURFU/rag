FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка Python-зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Загрузка моделей spaCy для русского языка 
RUN python -m spacy download ru_core_news_md

# Создание необходимых директорий
RUN mkdir -p indexes logs data

# Копирование кода приложения
COPY . .

# Порт для API
EXPOSE 8000

# Создаем скрипт запуска с инициализацией БД
RUN echo '#!/bin/sh\n\
echo "Initializing database..."\n\
python -m app.main\n' > /app/start.sh && \
chmod +x /app/start.sh

# Запуск скрипта
CMD ["/app/start.sh"] 