import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple

import ollama
from ollama import AsyncClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaLLM:
    """Класс для работы с моделями через Ollama."""
    
    def __init__(self, model_name: str = "mistral"):
        """
        Инициализация клиента Ollama.
        
        Args:
            model_name: Название модели в Ollama
        """
        self.model_name = model_name
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.base_url)
        self.async_client = AsyncClient(host=self.base_url)
        self._model_loaded = False
        logger.info(f"Initialized Ollama client for model {model_name}")
    
    async def ensure_model_loaded(self) -> bool:
        """
        Проверяет, загружена ли модель в Ollama, и при необходимости загружает ее.
        
        Returns:
            True, если модель успешно загружена
        """
        if self._model_loaded:
            return True
            
        try:
            # Проверяем, есть ли модель в списке доступных
            try:
                models_response = await self.async_client.list()
                logger.info(f"Response from Ollama API: {type(models_response)}")
                
                # Дебаг информация
                if isinstance(models_response, dict):
                    for key in models_response:
                        logger.info(f"Key in response: {key}, type: {type(models_response[key])}")
                        
                # Проверяем структуру ответа и извлекаем список моделей
                if not isinstance(models_response, dict):
                    logger.warning(f"Unexpected response from Ollama API: {models_response}")
                    models = []
                else:
                    models = models_response.get('models', [])
                    if not isinstance(models, list):
                        logger.warning(f"Unexpected models format: {models}")
                        models = []
                    else:
                        logger.info(f"Found {len(models)} models")
                        
                # Вывод информации о моделях для отладки
                for i, model in enumerate(models):
                    if isinstance(model, dict):
                        model_info = {key: value for key, value in model.items() 
                                     if key in ['name', 'model', 'modified_at', 'size']}
                        logger.info(f"Model {i}: {model_info}")
                        
                # Проверяем наличие модели по имени, учитывая возможные варианты представления имени
                model_name_parts = self.model_name.split(':')
                base_model_name = model_name_parts[0] if model_name_parts else ""
                
                model_exists = False
                for model in models:
                    if not isinstance(model, dict):
                        logger.warning(f"Unexpected model format: {model}")
                        continue
                        
                    # Попробуем все возможные ключи для имени модели
                    model_name = ""
                    for key in ['name', 'model', 'id']:
                        if key in model and model[key]:
                            model_name = model[key]
                            break
                        
                    if not model_name:
                        logger.warning(f"Could not find model name in: {model}")
                        continue
                        
                    logger.info(f"Checking model: {model_name} against {self.model_name} or {base_model_name}")
                    if model_name == self.model_name or model_name == base_model_name:
                        model_exists = True
                        logger.info(f"Model {self.model_name} found in available models")
                        break
                
                if not model_exists:
                    logger.info(f"Model {self.model_name} not found, pulling from Ollama library...")
                    # Начинаем загрузку модели асинхронно
                    await self.async_client.pull(self.model_name)
                    logger.info(f"Model {self.model_name} successfully pulled")
                else:
                    logger.info(f"Model {self.model_name} is already available")
                
                self._model_loaded = True
                return True
                
            except Exception as inner_e:
                logger.error(f"Error checking model availability: {str(inner_e)}")
                # Предполагаем, что модель не существует и пытаемся загрузить её
                try:
                    logger.info(f"Attempting to pull model {self.model_name} after error...")
                    await self.async_client.pull(self.model_name)
                    logger.info(f"Model {self.model_name} successfully pulled")
                    self._model_loaded = True
                    return True
                except Exception as pull_error:
                    logger.error(f"Error pulling model: {str(pull_error)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error ensuring model is loaded: {str(e)}")
            return False
    
    def ensure_model_loaded_sync(self) -> bool:
        """
        Синхронный вариант проверки наличия модели.
        
        Returns:
            True, если модель успешно загружена
        """
        if self._model_loaded:
            return True
            
        try:
            # Проверяем, есть ли модель в списке доступных
            models_response = self.client.list()
            
            # Проверяем структуру ответа и извлекаем список моделей
            if not isinstance(models_response, dict):
                logger.warning(f"Unexpected response from Ollama API: {models_response}")
                models = []
            else:
                models = models_response.get('models', [])
                if not isinstance(models, list):
                    logger.warning(f"Unexpected models format: {models}")
                    models = []
            
            # Проверяем наличие модели по имени, учитывая возможные варианты представления имени
            model_name_parts = self.model_name.split(':')
            base_model_name = model_name_parts[0] if model_name_parts else ""
            
            model_exists = False
            for model in models:
                if not isinstance(model, dict):
                    continue
                    
                model_name = model.get('name', '')
                if not model_name:
                    # Попробуем другие возможные ключи
                    model_name = model.get('model', '')
                    
                if model_name == self.model_name or model_name == base_model_name:
                    model_exists = True
                    break
            
            if not model_exists:
                logger.info(f"Model {self.model_name} not found, pulling from Ollama library...")
                # Загружаем модель
                for progress in self.client.pull(self.model_name):
                    if 'status' in progress:
                        logger.info(f"Pull status: {progress['status']}")
                logger.info(f"Model {self.model_name} successfully pulled")
            else:
                logger.info(f"Model {self.model_name} is already available")
            
            self._model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Error ensuring model is loaded: {str(e)}")
            return False
    
    async def check_model_availability(self) -> bool:
        """
        Проверяет доступность модели и сервера Ollama.
        
        Returns:
            True, если сервер Ollama доступен и модель загружена или может быть загружена
        """
        try:
            # Пробуем получить список моделей
            models_response = await self.async_client.list()
            
            # Если получили ответ, значит сервер доступен
            logger.info("Ollama server is available")
            
            # Проверяем наличие нашей модели
            if not isinstance(models_response, dict):
                logger.warning(f"Unexpected response from Ollama API: {models_response}")
                return True  # Сервер работает, даже если формат ответа не такой как ожидался
                
            models = models_response.get('models', [])
            if not isinstance(models, list):
                logger.warning(f"Unexpected models format: {models}")
                return True  # Сервер работает, формат не критичен для проверки доступности
            
            # Проверяем наличие модели по имени, учитывая возможные варианты представления имени
            model_name_parts = self.model_name.split(':')
            base_model_name = model_name_parts[0] if model_name_parts else ""
            
            for model in models:
                if not isinstance(model, dict):
                    continue
                
                # Пробуем разные ключи для извлечения имени модели
                model_name = model.get('name', '')
                if not model_name:
                    model_name = model.get('model', '')
                    if not model_name:
                        model_name = model.get('id', '')
                
                if model_name == self.model_name or model_name == base_model_name:
                    logger.info(f"Model {self.model_name} is available")
                    return True
            
            # Если модель не найдена, сервер все равно доступен
            logger.info(f"Model {self.model_name} is not available, but Ollama server is running")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {str(e)}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Асинхронная генерация ответа.
        
        Args:
            prompt: Входной промпт
            **kwargs: Дополнительные параметры для генерации
            
        Returns:
            Ответ модели
        """
        # Убеждаемся, что модель загружена
        if not self._model_loaded:
            loaded = await self.ensure_model_loaded()
            if not loaded:
                logger.warning(f"Could not load model {self.model_name}, trying to find any available model")
                
                # Если не удалось загрузить указанную модель, попробуем найти любую другую
                try:
                    models_response = await self.async_client.list()
                    if isinstance(models_response, dict) and 'models' in models_response:
                        models = models_response.get('models', [])
                        if models and isinstance(models, list) and len(models) > 0:
                            # Используем первую доступную модель
                            if isinstance(models[0], dict) and 'name' in models[0]:
                                self.model_name = models[0]['name']
                                logger.info(f"Falling back to available model: {self.model_name}")
                                self._model_loaded = True
                            else:
                                return "Ошибка: не удалось определить имя доступной модели"
                        else:
                            return "Ошибка: нет доступных моделей на сервере Ollama"
                    else:
                        return "Ошибка: некорректный ответ от API Ollama при попытке получить список моделей"
                except Exception as e:
                    logger.error(f"Error getting available models: {str(e)}")
                    return f"Ошибка: не удалось подключиться к Ollama. Пожалуйста, убедитесь, что Ollama установлена, запущена и доступна.\nПодробная инструкция:\n1. Скачайте Ollama с сайта https://ollama.com/download\n2. Установите и запустите приложение\n3. Убедитесь, что сервер работает на порту 11434\n4. Перезапустите приложение RAG\n\nТехническая информация об ошибке: {str(e)}"
        
        # Определяем количество повторных попыток
        max_retries = kwargs.pop('max_retries', 3)
        retry_count = 0
        last_error = None
        
        # Определяем, является ли промпт русскоязычным для специфичных настроек
        is_russian = True if "русск" in prompt.lower() or "КРАТКИЙ ОТВЕТ:" in prompt else False
        
        while retry_count < max_retries:
            try:
                # Оптимизированные параметры генерации для высокой answer_similarity
                options = {
                    'temperature': 0,     # Низкая температура для более детерминированных ответов
                    'num_predict': 2048,  # Увеличенный лимит токенов для полных ответов
                    'top_p': 0.95,        # Немного выше для учета большего числа вероятностей
                    'top_k': 20,          # Ограничиваем список кандидатов для каждого токена
                    'stop': ['<|im_end|>', '</answer>', '\n\nВОПРОС:', '\nВОПРОС:', 'ВОПРОС:']  # Расширенный список стоп-токенов
                }
                
                # Специальные настройки для русского языка
                if is_russian:
                    options['repeat_penalty'] = 1.15  # Немного выше для русского
                    options['frequency_penalty'] = 0.05  # Предотвращение повторов в русском
                
                # Извлекаем поддерживаемые параметры
                if 'temperature' in kwargs:
                    options['temperature'] = float(kwargs.pop('temperature'))
                if 'top_p' in kwargs:
                    options['top_p'] = float(kwargs.pop('top_p'))
                if 'num_predict' in kwargs:
                    options['num_predict'] = int(kwargs.pop('num_predict'))
                if 'stop' in kwargs and isinstance(kwargs['stop'], list):
                    options['stop'] = kwargs.pop('stop')
                if 'top_k' in kwargs:
                    options['top_k'] = int(kwargs.pop('top_k'))
                if 'repeat_penalty' in kwargs:
                    options['repeat_penalty'] = float(kwargs.pop('repeat_penalty'))
                if 'frequency_penalty' in kwargs:
                    options['frequency_penalty'] = float(kwargs.pop('frequency_penalty'))
                
                logger.info(f"Generating response with model {self.model_name}, options: {options}")
                
                # Вызываем генерацию с указанными параметрами
                response = await self.async_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
                
                # Упрощенная логика обработки ответа от Ollama:
                # 1. Если ответ - словарь и в нем есть ключ 'response', извлекаем его
                # 2. В противном случае пытаемся преобразовать ответ в строку и работать с ним
                if isinstance(response, dict) and 'response' in response:
                    response_text = response.get('response', '')
                else:
                    # Преобразуем ответ в строку
                    response_str = str(response)
                    # Ищем в строке подстроку "response='...'"
                    if "response='" in response_str:
                        start_idx = response_str.find("response='") + len("response='")
                        end_idx = response_str.find("'", start_idx)
                        if start_idx > 0 and end_idx > 0:
                            response_text = response_str[start_idx:end_idx]
                        else:
                            logger.warning(f"Не удалось извлечь текст ответа из строки: {response_str[:100]}")
                            response_text = ''
                    else:
                        logger.warning(f"Ответ имеет неизвестный формат: {response_str[:100]}")
                        response_text = ''
                
                if not response_text:
                    logger.warning(f"Empty response from Ollama: {response}")
                    retry_count += 1
                    last_error = "пустой ответ от модели"
                    await asyncio.sleep(1)
                    continue
                
                # Постобработка ответа для улучшения answer_similarity
                response_text = self._postprocess_response(response_text)
                
                return response_text
            
            except Exception as e:
                    # Отладочный вывод для анализа структуры ответа
                    logger.error(f"Ошибка при запросе к Ollama: {str(e)}")
                    retry_count += 1
                    last_error = str(e)
                    await asyncio.sleep(1)
                    continue

            # Успешный ответ, выходим из цикла
            break
        
        # Все попытки исчерпаны, но ответа нет
        if retry_count >= max_retries:
            return f"Ошибка генерации ответа после {max_retries} попыток: {last_error}"
    
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """
        Синхронный вариант генерации ответа.
        
        Args:
            prompt: Входной промпт
            **kwargs: Дополнительные параметры для генерации
            
        Returns:
            Ответ модели
        """
        # Убеждаемся, что модель загружена
        if not self._model_loaded:
            loaded = self.ensure_model_loaded_sync()
            if not loaded:
                logger.warning(f"Could not load model {self.model_name}, trying to find any available model")
                
                # Если не удалось загрузить указанную модель, попробуем найти любую другую
                try:
                    models_response = self.client.list()
                    if isinstance(models_response, dict) and 'models' in models_response:
                        models = models_response.get('models', [])
                        if models and isinstance(models, list) and len(models) > 0:
                            # Используем первую доступную модель
                            if isinstance(models[0], dict) and 'name' in models[0]:
                                self.model_name = models[0]['name']
                                logger.info(f"Falling back to available model: {self.model_name}")
                                self._model_loaded = True
                            else:
                                return "Ошибка: не удалось определить имя доступной модели"
                        else:
                            return "Ошибка: нет доступных моделей на сервере Ollama"
                    else:
                        return "Ошибка: некорректный ответ от API Ollama при попытке получить список моделей"
                except Exception as e:
                    logger.error(f"Error getting available models: {str(e)}")
                    return f"Ошибка: не удалось подключиться к Ollama. Пожалуйста, убедитесь, что Ollama установлена, запущена и доступна.\nПодробная инструкция:\n1. Скачайте Ollama с сайта https://ollama.com/download\n2. Установите и запустите приложение\n3. Убедитесь, что сервер работает на порту 11434\n4. Перезапустите приложение RAG\n\nТехническая информация об ошибке: {str(e)}"
        
        # Определяем количество повторных попыток
        max_retries = kwargs.pop('max_retries', 3)
        retry_count = 0
        last_error = None
        
        # Определяем, является ли промпт русскоязычным для специфичных настроек
        is_russian = True if "русск" in prompt.lower() or "КРАТКИЙ ОТВЕТ:" in prompt else False
        
        while retry_count < max_retries:
            try:
                # Оптимизированные параметры генерации для высокой answer_similarity
                options = {
                    'temperature': 0,     # Низкая температура для более детерминированных ответов
                    'num_predict': 2048,  # Увеличенный лимит токенов для полных ответов
                    'top_p': 0.95,        # Немного выше для учета большего числа вероятностей
                    'top_k': 20,          # Ограничиваем список кандидатов для каждого токена
                    'stop': ['<|im_end|>', '</answer>', '\n\nВОПРОС:', '\nВОПРОС:', 'ВОПРОС:']  # Расширенный список стоп-токенов
                }
                
                # Специальные настройки для русского языка
                if is_russian:
                    options['repeat_penalty'] = 1.15  # Немного выше для русского
                    options['frequency_penalty'] = 0.05  # Предотвращение повторов в русском
                
                # Извлекаем поддерживаемые параметры
                if 'temperature' in kwargs:
                    options['temperature'] = float(kwargs.pop('temperature'))
                if 'top_p' in kwargs:
                    options['top_p'] = float(kwargs.pop('top_p'))
                if 'num_predict' in kwargs:
                    options['num_predict'] = int(kwargs.pop('num_predict'))
                if 'stop' in kwargs and isinstance(kwargs['stop'], list):
                    options['stop'] = kwargs.pop('stop')
                if 'top_k' in kwargs:
                    options['top_k'] = int(kwargs.pop('top_k'))
                if 'repeat_penalty' in kwargs:
                    options['repeat_penalty'] = float(kwargs.pop('repeat_penalty'))
                if 'frequency_penalty' in kwargs:
                    options['frequency_penalty'] = float(kwargs.pop('frequency_penalty'))
                
                logger.info(f"Generating response with model {self.model_name}, options: {options}")
                
                # Вызываем генерацию с указанными параметрами
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
                
                # Упрощенная логика обработки ответа от Ollama:
                # 1. Если ответ - словарь и в нем есть ключ 'response', извлекаем его
                # 2. В противном случае пытаемся преобразовать ответ в строку и работать с ним
                if isinstance(response, dict) and 'response' in response:
                    response_text = response.get('response', '')
                else:
                    # Преобразуем ответ в строку
                    response_str = str(response)
                    # Ищем в строке подстроку "response='...'"
                    if "response='" in response_str:
                        start_idx = response_str.find("response='") + len("response='")
                        end_idx = response_str.find("'", start_idx)
                        if start_idx > 0 and end_idx > 0:
                            response_text = response_str[start_idx:end_idx]
                        else:
                            logger.warning(f"Не удалось извлечь текст ответа из строки: {response_str[:100]}")
                            response_text = ''
                    else:
                        logger.warning(f"Ответ имеет неизвестный формат: {response_str[:100]}")
                        response_text = ''
                
                if not response_text:
                    logger.warning(f"Empty response from Ollama: {response}")
                    retry_count += 1
                    last_error = "пустой ответ от модели"
                    time.sleep(1)
                    continue
                
                # Постобработка ответа для улучшения answer_similarity
                response_text = self._postprocess_response(response_text)
                
                return response_text
                
            except Exception as e:
                logger.error(f"Error generating response (attempt {retry_count+1}/{max_retries}): {str(e)}")
                retry_count += 1
                last_error = str(e)
                
                if retry_count < max_retries:
                    # Ждем перед следующей попыткой
                    time.sleep(1)
                    
                    # На второй попытке удаляем все параметры
                    if retry_count == 1:
                        logger.info("Retrying without advanced options")
                        kwargs = {}
                        
                    continue
        
        # Если все попытки неудачны, возвращаем ошибку
        return f"Ошибка генерации ответа: {last_error}"
    
    def _postprocess_response(self, response: str) -> str:
        """
        Постобработка ответа для улучшения метрики answer_similarity.
        
        Args:
            response: Исходный ответ модели
            
        Returns:
            Обработанный ответ
        """
        # Убираем вводные фразы, которые могут снизить metрику answer_similarity
        intro_phrases = [
            "Согласно предоставленному контексту,",
            "На основе предоставленной информации,",
            "Из контекста следует, что",
            "В соответствии с предоставленными данными,",
            "Судя по контексту,",
            "Как указано в контексте,",
            "Информация в контексте указывает, что",
            "Контекст свидетельствует о том, что",
            "Согласно информации,",
            "По имеющейся информации,",
            "Исходя из предоставленных материалов,"
        ]
        
        # Удаляем вводные фразы в начале ответа
        processed_response = response
        for phrase in intro_phrases:
            if processed_response.startswith(phrase):
                processed_response = processed_response[len(phrase):].lstrip()
                # После удаления одной фразы прекращаем, чтобы не удалить часть ответа
                break
        
        # Если ответ начинается со строчной буквы после удаления вводной фразы,
        # делаем первую букву заглавной
        if processed_response and processed_response[0].islower():
            processed_response = processed_response[0].upper() + processed_response[1:]
        
        # Удаляем лишние пробелы и переносы строк
        processed_response = processed_response.strip()
        
        # Удаляем точку в конце однострочного ответа для factoid-подобных вопросов,
        # если ответ короткий (менее 8 слов) и не содержит других предложений
        words = processed_response.split()
        if (len(words) < 8 and 
            processed_response.endswith('.') and 
            '.' not in processed_response[:-1] and 
            '!' not in processed_response and 
            '?' not in processed_response):
            processed_response = processed_response[:-1]
        
        return processed_response

# Глобальный экземпляр для использования в приложении
_ollama_instance = None

def get_ollama_instance(model_name: str = None, force_new: bool = False) -> OllamaLLM:
    """
    Создает или возвращает кешированный экземпляр OllamaLLM.
    
    Args:
        model_name: Имя модели для использования (если None, используется значение из .env)
        force_new: Принудительно создать новый экземпляр (при переключении модели)
    
    Returns:
        Экземпляр OllamaLLM с указанной моделью
    """
    global _ollama_instance
    
    if model_name is None:
        # Если модель не указана, используем значение из переменной окружения
        model_name = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
    
    # Проверяем, нужно ли создать новый экземпляр
    if _ollama_instance is None or force_new or _ollama_instance.model_name != model_name:
        # Создаем новый экземпляр
        _ollama_instance = OllamaLLM(model_name)
        logger.info(f"Created new OllamaLLM instance with model {model_name}")
    
    return _ollama_instance

async def preload_model():
    """
    Предварительно загружает модель при запуске приложения.
    """
    ollama = get_ollama_instance()
    logger.info("Preloading model...")
    await ollama.ensure_model_loaded()
    logger.info("Model preloaded successfully")
    
def preload_model_sync():
    """
    Синхронный вариант предварительной загрузки модели.
    """
    ollama = get_ollama_instance()
    logger.info("Preloading model...")
    ollama.ensure_model_loaded_sync()
    logger.info("Model preloaded successfully") 