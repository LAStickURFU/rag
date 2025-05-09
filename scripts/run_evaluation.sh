#!/bin/bash
# Скрипт для запуска полного цикла оценки качества RAG-системы

# Проверяем наличие Qdrant и Ollama
echo "Проверка статуса сервисов..."
curl -s http://localhost:6333/health > /dev/null
if [ $? -ne 0 ]; then
    echo "ОШИБКА: Qdrant не запущен. Запустите его с помощью docker-compose up -d qdrant"
    exit 1
fi

curl -s http://localhost:11434/api/tags > /dev/null
if [ $? -ne 0 ]; then
    echo "ОШИБКА: Ollama не запущен. Запустите его и убедитесь, что он доступен."
    exit 1
fi

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Параметры
COLLECTION_NAME="test_documents"
DATASET_NAME=""
LIMIT=10
BUILTIN=false

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --collection=*)
        COLLECTION_NAME="${1#*=}"
        shift
        ;;
        --dataset=*)
        DATASET_NAME="${1#*=}"
        BUILTIN=true
        shift
        ;;
        --limit=*)
        LIMIT="${1#*=}"
        shift
        ;;
        --help)
        echo "Использование: $0 [--collection=ИМЯ_КОЛЛЕКЦИИ] [--dataset=ИМЯ_НАБОРА_ДАННЫХ] [--limit=КОЛИЧЕСТВО]"
        echo "  --collection=ИМЯ_КОЛЛЕКЦИИ  Имя коллекции в Qdrant (по умолчанию 'test_documents')"
        echo "  --dataset=ИМЯ_НАБОРА_ДАННЫХ Использовать встроенный набор данных (sberquad, RuBQ)"
        echo "  --limit=КОЛИЧЕСТВО          Ограничить количество примеров для оценки"
        exit 0
        ;;
        *)
        echo "Неизвестный параметр: $1"
        echo "Используйте --help для получения справки"
        exit 1
        ;;
    esac
done

echo -e "${YELLOW}Запуск полного цикла оценки качества RAG-системы${NC}"
echo "Параметры:"
echo "  Коллекция: $COLLECTION_NAME"
if [ "$BUILTIN" = true ]; then
    echo "  Набор данных: $DATASET_NAME"
else
    echo "  Набор данных: тестовый"
fi
echo "  Количество примеров: $LIMIT"

# Создаем директории для результатов, если их нет
mkdir -p app/evaluation/results
mkdir -p app/evaluation/error_logs

echo -e "\n${YELLOW}[1/3] Подготовка тестовых данных...${NC}"
python3 scripts/prepare_test_data.py --collection="$COLLECTION_NAME"
if [ $? -ne 0 ]; then
    echo -e "${RED}Ошибка при подготовке тестовых данных${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[2/3] Запуск оценки качества...${NC}"
EVAL_CMD="python3 test_rag_eval.py --limit=$LIMIT"
if [ "$BUILTIN" = true ]; then
    EVAL_CMD="$EVAL_CMD --builtin --dataset=$DATASET_NAME"
fi
eval $EVAL_CMD

if [ $? -ne 0 ]; then
    echo -e "${RED}Ошибка при оценке качества${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[3/3] Анализ результатов...${NC}"
# Находим последний файл с результатами
RESULTS_FILE=$(ls -t app/evaluation/results/*.json | head -1)
if [ -z "$RESULTS_FILE" ]; then
    echo -e "${RED}Не найден файл с результатами оценки${NC}"
    exit 1
fi

echo -e "Файл с результатами: ${GREEN}$RESULTS_FILE${NC}"

# Краткая сводка результатов
echo -e "\n${YELLOW}Сводка результатов:${NC}"
python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
    metrics = data.get('aggregated_metrics', {})
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f'{metric}: {value:.4f}')
        else:
            print(f'{metric}: {value}')
    
    # Анализ качества
    answer_sim = metrics.get('answer_similarity_avg', 0)
    context_prec = metrics.get('context_precision_avg', 0)
    low_percentage = metrics.get('low_similarity_percentage', 100)
    
    print('\nАнализ качества:')
    if answer_sim > 0.3:
        print('\033[0;32m✓ Хорошее сходство ответов (> 0.3)\033[0m')
    elif answer_sim > 0.1:
        print('\033[1;33m⚠ Среднее сходство ответов (0.1-0.3)\033[0m')
    else:
        print('\033[0;31m✗ Низкое сходство ответов (< 0.1)\033[0m')
        
    if context_prec > 0.7:
        print('\033[0;32m✓ Высокая точность поиска контекста (> 0.7)\033[0m')
    elif context_prec > 0.3:
        print('\033[1;33m⚠ Средняя точность поиска контекста (0.3-0.7)\033[0m')
    else:
        print('\033[0;31m✗ Низкая точность поиска контекста (< 0.3)\033[0m')
        
    if low_percentage < 50:
        print('\033[0;32m✓ Низкий процент ответов с низким сходством (< 50%)\033[0m')
    elif low_percentage < 80:
        print('\033[1;33m⚠ Средний процент ответов с низким сходством (50-80%)\033[0m')
    else:
        print('\033[0;31m✗ Высокий процент ответов с низким сходством (> 80%)\033[0m')
"

echo -e "\n${GREEN}Оценка качества RAG-системы завершена!${NC}"
echo "Для улучшения результатов рекомендуется:"
echo "1. Проверить правильность работы токенового чанкера"
echo "2. Настроить параметры гибридного поиска (dense_weight, sparse_weight)"
echo "3. Проверить статус векторного хранилища Qdrant и доступность документов"
echo "4. Использовать более подходящую модель для эмбеддингов (например, ai-forever/sbert_large_nlu_ru)"

exit 0 