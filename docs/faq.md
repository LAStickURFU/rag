# Часто задаваемые вопросы (FAQ)

## Общие вопросы

### Что такое RAG?

RAG (Retrieval-Augmented Generation) - это подход, который объединяет поиск информации из базы знаний с генеративными возможностями языковых моделей. Система сначала ищет релевантные фрагменты в вашей базе документов, а затем использует найденную информацию для формирования более точного и фактически корректного ответа.

### Чем RAG отличается от обычного использования LLM?

Обычные LLM генерируют ответы исключительно на основе знаний, полученных во время обучения. Это имеет несколько ограничений:
- Знания статичны и могут устаревать
- Отсутствует доступ к специфической информации из ваших документов
- Высокая вероятность галлюцинаций и неточностей

RAG решает эти проблемы, дополняя знания модели актуальной информацией из ваших документов, что позволяет получать более точные и достоверные ответы.

### Какие преимущества дает RAG?

- **Повышение точности ответов**: ответы основаны на актуальной информации из ваших документов
- **Снижение галлюцинаций**: модель опирается на фактическую информацию
- **Прозрачность источников**: возможность увидеть, откуда взята информация для ответа
- **Адаптивность к специфическим данным**: работа с вашими собственными документами без дообучения модели
- **Конфиденциальность**: ваши данные не передаются во внешние сервисы

## Технические вопросы

### Какие модели используются в системе?

Система использует несколько типов моделей:
- **Языковая модель**: mistral:7b-instruct (через Ollama) для генерации ответов
- **Модель эмбеддингов**: intfloat/multilingual-e5-large для векторизации текста
- **Cross-encoder**: cross-encoder/ms-marco-MiniLM-L-6-v2 для переранжирования результатов поиска

### Какие форматы документов поддерживаются?

Система поддерживает множество форматов:
- PDF (.pdf)
- Microsoft Word (.docx, .doc)
- Текстовые файлы (.txt)
- Markdown (.md)
- JSON (.json)
- HTML (.html, .htm)
- и другие текстовые форматы

### Как выбрать оптимальный метод чанкинга?

Выбор метода чанкинга зависит от типа ваших документов:
- **Символьный чанкинг** (character): универсальный подход, подходит для большинства текстов
- **Токеновый чанкинг** (token): учитывает токенизацию LLM, полезен для специализированных текстов
- **Семантический чанкинг** (semantic): учитывает смысловые единицы, полезен для сложных текстов
- **Иерархический чанкинг** (hierarchical): сохраняет структуру документа, отлично подходит для структурированных текстов с заголовками

### Как работает гибридный поиск?

Гибридный поиск комбинирует результаты двух подходов:
1. **Dense Retrieval** (векторный поиск): находит семантически похожие фрагменты
2. **Sparse Retrieval** (BM25): находит фрагменты с точными совпадениями ключевых слов

Результаты комбинируются с учетом заданных весов (по умолчанию: dense_weight=0.3, sparse_weight=0.7) и затем переранжируются с помощью cross-encoder для получения наиболее релевантных фрагментов.

## Масштабирование и производительность

### Какое количество документов может обрабатывать система?

Система способна работать с тысячами документов. Ограничения зависят от:
- Доступной оперативной памяти (для индексации и поиска)
- Дискового пространства (для хранения векторов и метаданных)
- Требований к скорости ответа

Для очень больших коллекций рекомендуется использовать более мощное оборудование или настроить разделение на несколько коллекций.

### Как оптимизировать скорость работы?

Для оптимизации скорости:
- Используйте более мощные GPU для ускорения векторизации и генерации
- Настройте параметры чанкинга (увеличение размера чанков сокращает их количество)
- Используйте менее ресурсоемкие модели эмбеддингов (например, e5-small вместо e5-large)
- Отключите переранжирование для очень больших коллекций, если приоритет - скорость
- Используйте кэширование эмбеддингов для часто используемых документов

### Можно ли использовать GPU для ускорения?

Да, система поддерживает использование GPU для:
1. Создания эмбеддингов (векторизации)
2. Работы с языковыми моделями через Ollama

Для включения GPU-поддержки:
- Убедитесь, что у вас установлены CUDA и соответствующие драйверы
- Настройте переменные окружения в файле `.env` (USE_GPU=true)
- В docker-compose.yml добавьте настройки GPU для контейнера backend

## Настройка и интеграция

### Как изменить языковую модель?

Для изменения языковой модели:
1. Убедитесь, что нужная модель загружена в Ollama: `ollama pull название-модели`
2. Измените значение `OLLAMA_MODEL` в файле `.env` или в настройках пользователя через веб-интерфейс

### Можно ли использовать собственные модели эмбеддингов?

Да, система поддерживает использование любых моделей эмбеддингов из Hugging Face:
1. Установите желаемую модель в `.env`: `EMBEDDING_MODEL=название/модели`
2. Перезапустите сервис backend или выполните переиндексацию документов

### Как интегрировать систему с другими приложениями?

Система предоставляет RESTful API для интеграции:
- Используйте эндпоинт `/token` для аутентификации
- Используйте `/ask` для запросов с RAG
- Используйте `/direct-ask` для прямых запросов к LLM
- Используйте другие эндпоинты для управления документами и настройками

Полная документация API доступна в [API Reference](./api-reference.md).

## Устранение неполадок

### Ответы не содержат информацию из моих документов

Возможные причины и решения:
- **Документы не проиндексированы**: проверьте статус в разделе "Документы"
- **Запрос сформулирован неоптимально**: используйте ключевые слова из ваших документов
- **Недостаточно фрагментов в контексте**: увеличьте параметр top_k_chunks в настройках
- **Неподходящее разбиение на фрагменты**: попробуйте другой метод чанкинга

### Система выдает ошибку при загрузке документов

Проверьте следующее:
- Формат документа поддерживается системой
- Размер документа не превышает ограничений (обычно до 50 МБ)
- Документ не поврежден и может быть открыт другими программами
- Достаточно дискового пространства для индексации

### Ollama недоступен или не отвечает

Если система не может подключиться к Ollama:
1. Убедитесь, что Ollama запущен: `ollama serve`
2. Проверьте, что модель загружена: `ollama list`
3. Проверьте настройки подключения в `.env` (OLLAMA_HOST)
4. При использовании Docker убедитесь, что настроен корректный host.docker.internal

### Как очистить все данные и начать заново?

Для полной очистки системы:
1. Остановите все контейнеры: `docker-compose down -v`
2. Удалите директории с данными: `rm -rf ./data`
3. Повторно запустите систему: `./setup.sh` 