# API Reference

В этом документе описаны все API-эндпоинты RAG-системы, доступные через REST API.

## Базовая информация

- **Базовый URL**: `http://localhost:8000`
- **Формат ответов**: JSON
- **Аутентификация**: JWT-токены (Bearer Authentication)

## Аутентификация и авторизация

### Получение токена доступа

```
POST /token
```

**Тело запроса**:
```json
{
  "username": "string",
  "password": "string"
}
```

**Ответ** (200 OK):
```json
{
  "access_token": "string",
  "token_type": "bearer"
}
```

**Ошибки**:
- 401 Unauthorized - неверные учетные данные

### Регистрация нового пользователя

```
POST /register
```

**Тело запроса**:
```json
{
  "username": "string",
  "password": "string",
  "email": "string"
}
```

**Ответ** (201 Created):
```json
{
  "id": 0,
  "username": "string",
  "email": "string",
  "role": "user",
  "created_at": "2023-01-01T00:00:00.000Z"
}
```

**Ошибки**:
- 400 Bad Request - пользователь с таким именем уже существует

### Получение данных текущего пользователя

```
GET /me
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "username": "string",
  "email": "string",
  "role": "string",
  "created_at": "2023-01-01T00:00:00.000Z"
}
```

## Работа с документами

### Загрузка документов

```
POST /documents/upload
```

**Заголовки**:
```
Authorization: Bearer {token}
Content-Type: multipart/form-data
```

**Параметры формы**:
- `files`: файлы для загрузки (PDF, DOCX, TXT, MD, JSON и др.)
- `titles`: названия для файлов (в том же порядке, что и файлы)

**Ответ** (200 OK):
```json
{
  "status": "success",
  "documents": [
    {
      "id": 0,
      "title": "string",
      "filename": "string",
      "size": 0,
      "status": "uploaded",
      "uuid": "string"
    }
  ],
  "message": "Документы приняты на обработку"
}
```

**Ошибки**:
- 400 Bad Request - количество файлов и заголовков не совпадает
- 401 Unauthorized - отсутствует токен аутентификации

### Получение списка документов

```
GET /documents
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Параметры запроса**:
- `all_users` (boolean, опционально): получить документы всех пользователей (только для админов)

**Ответ** (200 OK):
```json
[
  {
    "id": 0,
    "title": "string",
    "filename": "string",
    "size": 0,
    "chunks_count": 0,
    "status": "string",
    "created_at": "2023-01-01T00:00:00.000Z",
    "uuid": "string",
    "user_id": 0,
    "uploader": "string",
    "chunking_mode": "string",
    "embedding_model": "string",
    "processing_summary": {
      "chunk_size": 0,
      "chunk_overlap": 0,
      "has_additional_params": true
    }
  }
]
```

### Получение информации о документе

```
GET /documents/{document_id}
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "title": "string",
  "content": "string",
  "source": "string",
  "user_id": 0,
  "created_at": "2023-01-01T00:00:00.000Z",
  "status": "string",
  "file_name": "string",
  "file_size": 0,
  "chunks_count": 0,
  "uuid": "string",
  "chunking_mode": "string",
  "chunk_size": 0,
  "chunk_overlap": 0,
  "embedding_model": "string",
  "processing_params": {}
}
```

**Ошибки**:
- 404 Not Found - документ не найден
- 403 Forbidden - нет доступа к документу

### Удаление документа

```
DELETE /documents/{document_id}
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "status": "success",
  "message": "Документ удален"
}
```

**Ошибки**:
- 404 Not Found - документ не найден
- 403 Forbidden - нет доступа к документу

### Переиндексация документов

```
POST /documents/reindex
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Параметры запроса**:
- `all_users` (boolean, опционально): переиндексировать документы всех пользователей (только для админов)

**Ответ** (200 OK):
```json
{
  "status": "success",
  "message": "Запущена переиндексация X документов. Обработка продолжается в фоне.",
  "documents_count": 0
}
```

### Очистка индекса

```
POST /index/clear
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Параметры запроса**:
- `user_id` (integer, опционально): ID пользователя, чей индекс нужно очистить (только для админов)

**Ответ** (200 OK):
```json
{
  "status": "success",
  "message": "Индекс очищен"
}
```

## Чат и генерация ответов

### Запрос с использованием RAG

```
POST /ask
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Тело запроса**:
```json
{
  "question": "string"
}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "user_id": 0,
  "question": "string",
  "response": "string",
  "created_at": "2023-01-01T00:00:00.000Z",
  "relevant_chunks": [
    {
      "text": "string",
      "relevance": 0.95,
      "doc_id": "string",
      "chunk_id": 0,
      "metadata": {
        "title": "string",
        "source": "string"
      }
    }
  ],
  "meta": {
    "retrieval_time_ms": 0,
    "generation_time_ms": 0,
    "retrieved_chunks_count": 0,
    "prompt_length": 0,
    "response_length": 0,
    "model": "string",
    "temperature": 0,
    "top_k_chunks": 0,
    "trace_id": "string",
    "timestamp": "2023-01-01T00:00:00.000Z",
    "rag_used": true
  },
  "rag_used": true
}
```

### Прямой запрос к модели (без RAG)

```
POST /direct-ask
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Тело запроса**:
```json
{
  "question": "string"
}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "user_id": 0,
  "question": "string",
  "response": "string",
  "created_at": "2023-01-01T00:00:00.000Z",
  "meta": {
    "generation_time_ms": 0,
    "model": "string",
    "temperature": 0,
    "trace_id": "string",
    "timestamp": "2023-01-01T00:00:00.000Z",
    "rag_used": false
  },
  "rag_used": false
}
```

### Получение истории чатов

```
GET /chats
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
[
  {
    "id": 0,
    "user_id": 0,
    "question": "string",
    "response": "string",
    "created_at": "2023-01-01T00:00:00.000Z",
    "relevant_chunks": [],
    "meta": {},
    "rag_used": true
  }
]
```

### Очистка истории чатов

```
DELETE /chats/clear
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "status": "success",
  "message": "История чата очищена"
}
```

## Настройки модели

### Получение настроек модели

```
GET /model/settings
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "user_id": 0,
  "temperature": 0,
  "top_p": 0,
  "max_tokens": 0,
  "top_k_chunks": 0,
  "context_window": 0,
  "model_name": "string",
  "created_at": "2023-01-01T00:00:00.000Z",
  "updated_at": "2023-01-01T00:00:00.000Z"
}
```

### Обновление настроек модели

```
POST /model/settings
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Тело запроса**:
```json
{
  "temperature": 0,
  "top_p": 0,
  "max_tokens": 0,
  "top_k_chunks": 0,
  "context_window": 0,
  "model_name": "string"
}
```

**Ответ** (200 OK):
```json
{
  "id": 0,
  "user_id": 0,
  "temperature": 0,
  "top_p": 0,
  "max_tokens": 0,
  "top_k_chunks": 0,
  "context_window": 0,
  "model_name": "string",
  "created_at": "2023-01-01T00:00:00.000Z",
  "updated_at": "2023-01-01T00:00:00.000Z"
}
```

### Проверка доступных моделей

```
GET /model/available
```

**Заголовки**:
```
Authorization: Bearer {token}
```

**Ответ** (200 OK):
```json
{
  "models": [
    {
      "name": "string",
      "size": 0,
      "quantization": "string",
      "format": "string"
    }
  ],
  "default_model": "string",
  "current_model": "string"
}
```

## Системные эндпоинты

### Проверка состояния системы

```
GET /api/healthcheck
```

**Ответ** (200 OK):
```json
{
  "status": "ok",
  "timestamp": "2023-01-01T00:00:00.000Z",
  "components": {
    "rag_service": "available",
    "ollama": "available",
    "ollama_model": "string"
  },
  "environment": {
    "embedding_model": "string",
    "spacy_model": "string",
    "chunk_size": 0,
    "chunk_overlap": 0,
    "python_version": "string",
    "pytorch_version": "string"
  }
}
```

## Коды ответов

- **200 OK** - запрос успешно обработан
- **201 Created** - ресурс успешно создан
- **400 Bad Request** - ошибка в параметрах запроса
- **401 Unauthorized** - отсутствует или недействителен токен аутентификации
- **403 Forbidden** - недостаточно прав для выполнения операции
- **404 Not Found** - запрашиваемый ресурс не найден
- **500 Internal Server Error** - внутренняя ошибка сервера

## Примеры использования API

### Авторизация и запрос

```python
import requests

# Получение токена
auth_response = requests.post(
    "http://localhost:8000/token",
    data={"username": "admin", "password": "admin123"}
)
token = auth_response.json()["access_token"]

# Запрос с использованием RAG
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Что такое RAG?"},
    headers=headers
)

print(response.json()["response"])
print("Источники:")
for chunk in response.json()["relevant_chunks"]:
    print(f"- {chunk['text'][:100]}... (релевантность: {chunk['relevance']})")
```

### Загрузка документа и получение списка

```python
import requests

# Получение токена
auth_response = requests.post(
    "http://localhost:8000/token",
    data={"username": "admin", "password": "admin123"}
)
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Загрузка документа
with open("document.txt", "rb") as f:
    files = {"files": ("document.txt", f)}
    data = {"titles": "Мой документ"}
    upload_response = requests.post(
        "http://localhost:8000/documents/upload",
        files=files,
        data=data,
        headers=headers
    )

# Получение списка документов
documents_response = requests.get(
    "http://localhost:8000/documents",
    headers=headers
)

for doc in documents_response.json():
    print(f"{doc['title']} - {doc['status']}")
``` 