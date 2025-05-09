"""
Тестирование улучшенного RobustChunker с разными типами контента.
"""

import json
import os
from app.chunking.robust_chunker import RobustChunker, Document

def print_separator(title):
    """Выводит разделитель с заголовком."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def test_json_chunking():
    """Тестирование разбиения JSON."""
    print_separator("JSON CHUNKING TEST")
    
    # Создаем JSON для тестирования
    test_json = {
        "items": [
            {"id": 1, "name": "Item 1", "description": "This is a test item with index 1"},
            {"id": 2, "name": "Item 2", "description": "This is a test item with index 2"},
            {"id": 3, "name": "Item 3", "description": "This is a test item with index 3"},
            {"id": 4, "name": "Item 4", "description": "This is a test item with index 4"},
            {"id": 5, "name": "Item 5", "description": "This is a test item with index 5"}
        ],
        "metadata": {
            "total_items": 5,
            "source": "test",
            "created_at": "2023-05-08T12:00:00Z",
            "description": "This is a test JSON document with multiple items"
        }
    }
    
    json_str = json.dumps(test_json, indent=2)
    
    # Создаем экземпляр RobustChunker с небольшим размером чанка для тестирования
    chunker = RobustChunker(chunk_size=200, chunk_overlap=50)
    
    # Создаем документ
    doc = Document(content=json_str, metadata={"source": "test.json"})
    
    # Разбиваем на чанки
    chunks = chunker.create_chunks_from_document(doc, "json-test")
    
    # Выводим результаты
    print(f"JSON document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text ({len(chunk.text)} chars):")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print(f"Metadata: {chunk.metadata}")

def test_csv_chunking():
    """Тестирование разбиения CSV."""
    print_separator("CSV CHUNKING TEST")
    
    # Создаем CSV для тестирования
    csv_data = "id,name,email,age,city\n"
    for i in range(1, 21):
        csv_data += f"{i},User {i},user{i}@example.com,{20+i},City {i}\n"
    
    # Создаем экземпляр RobustChunker с небольшим размером чанка для тестирования
    chunker = RobustChunker(chunk_size=200, chunk_overlap=50)
    
    # Создаем документ
    doc = Document(content=csv_data, metadata={"source": "test.csv"})
    
    # Разбиваем на чанки
    chunks = chunker.create_chunks_from_document(doc, "csv-test")
    
    # Выводим результаты
    print(f"CSV document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text ({len(chunk.text)} chars):")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print(f"Metadata: {chunk.metadata}")

def test_xml_chunking():
    """Тестирование разбиения XML."""
    print_separator("XML CHUNKING TEST")
    
    # Создаем XML для тестирования
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
  <book id="1">
    <title>Book Title 1</title>
    <author>Author 1</author>
    <year>2020</year>
    <description>This is a description of book 1</description>
  </book>
  <book id="2">
    <title>Book Title 2</title>
    <author>Author 2</author>
    <year>2021</year>
    <description>This is a description of book 2</description>
  </book>
  <book id="3">
    <title>Book Title 3</title>
    <author>Author 3</author>
    <year>2022</year>
    <description>This is a description of book 3</description>
  </book>
</catalog>
"""
    
    # Создаем экземпляр RobustChunker с небольшим размером чанка для тестирования
    chunker = RobustChunker(chunk_size=200, chunk_overlap=50)
    
    # Создаем документ
    doc = Document(content=xml_data, metadata={"source": "test.xml"})
    
    # Разбиваем на чанки
    chunks = chunker.create_chunks_from_document(doc, "xml-test")
    
    # Выводим результаты
    print(f"XML document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text ({len(chunk.text)} chars):")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print(f"Metadata: {chunk.metadata}")

def test_yaml_chunking():
    """Тестирование разбиения YAML."""
    print_separator("YAML CHUNKING TEST")
    
    # Создаем YAML для тестирования
    yaml_data = """
# This is a test YAML file
version: 1.0
app:
  name: TestApp
  description: This is a test YAML document
  settings:
    debug: true
    log_level: info
    timeout: 30

users:
  - name: User 1
    email: user1@example.com
    roles:
      - admin
      - editor
  - name: User 2
    email: user2@example.com
    roles:
      - user
  - name: User 3
    email: user3@example.com
    roles:
      - user
      - editor
"""
    
    # Создаем экземпляр RobustChunker с небольшим размером чанка для тестирования
    chunker = RobustChunker(chunk_size=200, chunk_overlap=50)
    
    # Создаем документ
    doc = Document(content=yaml_data, metadata={"source": "test.yaml"})
    
    # Разбиваем на чанки
    chunks = chunker.create_chunks_from_document(doc, "yaml-test")
    
    # Выводим результаты
    print(f"YAML document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text ({len(chunk.text)} chars):")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print(f"Metadata: {chunk.metadata}")

def test_txt_chunking():
    """Тестирование разбиения обычного текста."""
    print_separator("TXT CHUNKING TEST")
    
    # Создаем текст для тестирования
    txt_data = """Это тестовый текстовый файл для проверки работы улучшенного RobustChunker.
    
В этом файле содержится несколько абзацев простого текста.

Абзац 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor, nisl eget ultricies lacinia, nisl nisl aliquam nisl, eget ultricies nisl nisl eget nisl.

Абзац 2: Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.

Абзац 3: Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.
"""
    
    # Создаем экземпляр RobustChunker с небольшим размером чанка для тестирования
    chunker = RobustChunker(chunk_size=200, chunk_overlap=50)
    
    # Создаем документ
    doc = Document(content=txt_data, metadata={"source": "test.txt"})
    
    # Разбиваем на чанки
    chunks = chunker.create_chunks_from_document(doc, "txt-test")
    
    # Выводим результаты
    print(f"TXT document split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text ({len(chunk.text)} chars):")
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
        print(f"Metadata: {chunk.metadata}")

def main():
    """Основная функция тестирования."""
    # Тестируем разбиение разных типов контента
    test_json_chunking()
    test_csv_chunking()
    test_xml_chunking()
    test_yaml_chunking()
    test_txt_chunking()

if __name__ == "__main__":
    main() 