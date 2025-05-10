"""
Модуль для извлечения текста из документов различных форматов.
"""

import json
import yaml
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
from odf.opendocument import load as odf_load
from odf.text import P
from ebooklib import epub

def extract_text(file, contents: bytes, file_name=None) -> str:
    """
    Извлекает текст из файла различных форматов.
    
    Args:
        file: Объект файла с атрибутом filename или None
        contents: Содержимое файла в байтах
        file_name: Имя файла (используется, если file is None)
    
    Returns:
        Текстовое содержимое файла
    """
    import traceback
    
    # Определяем имя файла
    filename = None
    if file and hasattr(file, 'filename'):
        filename = file.filename
    elif file and hasattr(file, 'name'):
        filename = file.name
    elif file_name:
        filename = file_name
    else:
        raise ValueError("Не удалось определить имя файла")
    
    # Получаем расширение
    ext = filename.lower().split('.')[-1]
    
    if ext in ['txt', 'md', 'log', 'tex', 'jsonl']:
        return contents.decode('utf-8', errors='ignore')
    elif ext == 'pdf':
        try:
            pdf_file = BytesIO(contents)
            reader = PdfReader(pdf_file)
            
            # Проверяем, есть ли страницы в документе
            if len(reader.pages) == 0:
                return "PDF документ не содержит страниц"
            
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ''
                    text_parts.append(page_text)
                except Exception as e:
                    # Логируем ошибку конкретной страницы, но продолжаем обработку
                    text_parts.append(f"[Ошибка извлечения текста со страницы {i+1}]")
                    print(f"Ошибка при чтении страницы {i+1} PDF: {str(e)}")
            
            return '\n'.join(text_parts)
        except Exception as e:
            # Подробно логируем ошибку для отладки
            error_details = traceback.format_exc()
            print(f"Ошибка при обработке PDF {filename}: {str(e)}\n{error_details}")
            raise Exception(f"Ошибка при чтении PDF: {str(e)}")
    elif ext in ['doc', 'docx']:
        doc = DocxDocument(BytesIO(contents))
        return '\n'.join([p.text for p in doc.paragraphs])
    elif ext == 'rtf':
        return rtf_to_text(contents.decode('utf-8', errors='ignore'))
    elif ext in ['json']:
        try:
            obj = json.loads(contents.decode('utf-8', errors='ignore'))
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return contents.decode('utf-8', errors='ignore')
    elif ext in ['yaml', 'yml']:
        try:
            obj = yaml.safe_load(contents.decode('utf-8', errors='ignore'))
            return yaml.dump(obj, allow_unicode=True)
        except Exception:
            return contents.decode('utf-8', errors='ignore')
    elif ext in ['csv', 'tsv']:
        return contents.decode('utf-8', errors='ignore')
    elif ext == 'html' or ext == 'htm':
        soup = BeautifulSoup(contents, 'html.parser')
        return soup.get_text(separator='\n', strip=True)
    elif ext == 'epub':
        try:
            book = epub.read_epub(BytesIO(contents))
            text = []
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text.append(soup.get_text(separator='\n', strip=True))
            return '\n\n'.join(text)
        except Exception as e:
            print(f"Ошибка при обработке EPUB {filename}: {str(e)}")
            raise Exception(f"Ошибка при чтении EPUB: {str(e)}")
    elif ext == 'odt':
        try:
            textdoc = odf_load(BytesIO(contents))
            text = []
            for paragraph in textdoc.getElementsByType(P):
                text.append(paragraph.firstChild.data if paragraph.firstChild else '')
            return '\n'.join(text)
        except Exception as e:
            print(f"Ошибка при обработке ODT {filename}: {str(e)}")
            raise Exception(f"Ошибка при чтении ODT: {str(e)}")
    else:
        return contents.decode('utf-8', errors='ignore') 