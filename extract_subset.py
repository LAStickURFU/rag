#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

# Пути к файлам с учетом новой структуры
input_file = "evaluation/test_dataset.json"
output_file = "evaluation/test_dataset_subset.json"

# Создаем абсолютные пути от корня проекта
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
input_file_path = os.path.join(root_dir, input_file)
output_file_path = os.path.join(root_dir, output_file)

# Загрузка полного датасета
with open(input_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Вывод статистики
print(f"Всего элементов в исходном датасете: {len(data)}")

# Берем первые 10 элементов
subset = data[:10]

# Сохраняем подмножество
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False, indent=2)

print(f"Сохранено {len(subset)} элементов в файл {output_file}") 