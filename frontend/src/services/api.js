import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Создание настроенного экземпляра axios
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Добавление перехватчика для установки токена авторизации
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Создание пользователя
export const registerUser = async (username, password) => {
  try {
    const response = await api.post('/register', { username, password });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка регистрации');
  }
};

// Получение токена доступа
export const getToken = async (username, password) => {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_URL}/token`, {
      method: 'POST',
      body: formData
    });
   
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Ошибка авторизации');
    }
   
    return await response.json();
  } catch (error) {
    console.error('Ошибка при получении токена:', error);
    throw error;
  }
};

// Выход пользователя
export const logout = () => {
  localStorage.removeItem('token');
};

// Проверка авторизации
export const isAuthenticated = () => {
  return localStorage.getItem('token') !== null;
};

// Прямой запрос к LLM без использования RAG
export const askDirectQuestion = async (question) => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('Требуется авторизация');
    }

    const response = await api.post('/direct-ask', { question });
    return response.data;
  } catch (error) {
    console.error('Ошибка при отправке прямого запроса:', error);
    throw error.response ? error.response.data : new Error('Ошибка при отправке запроса');
  }
};

// Запрос с использованием RAG и автоматическим переходом на прямой запрос при ошибке
export const askQuestion = async (question) => {
  try {
    const response = await api.post('/ask', { question });
    return response.data;
  } catch (error) {
    console.error('Ошибка при отправке запроса:', error);
   
    // Если ошибка связана с RAG, используем прямой запрос
    if (error.response &&
        error.response.data &&
        error.response.data.detail &&
        error.response.data.detail.includes('Input/output error')) {
      console.warn('Ошибка RAG, переключение на прямой запрос к LLM');
      return await askDirectQuestion(question);
    }
   
    throw error.response ? error.response.data : new Error('Ошибка при отправке запроса');
  }
};

// Получение истории чата
export const getChatHistory = async () => {
  try {
    const response = await api.get('/chats');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении истории чата');
  }
};

// Загрузка документов (мультифайл)
export const uploadDocuments = async (titles, files) => {
  try {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    titles.forEach(title => formData.append('titles', title));
    const response = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при загрузке документов');
  }
};

// Получение списка документов
export const getDocuments = async () => {
  try {
    const response = await api.get('/documents');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении списка документов');
  }
};

// Переиндексация всех документов пользователя
export const reindexDocuments = async () => {
  try {
    const response = await api.post('/documents/reindex');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при переиндексации документов');
  }
};

// Получение настроек модели
export const getModelSettings = async () => {
  try {
    const response = await api.get('/model/settings');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении настроек модели');
  }
};

// Обновление настроек модели
export const updateModelSettings = async (settings) => {
  try {
    const response = await api.post('/model/settings', settings);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при обновлении настроек модели');
  }
};

// Функция для оценки качества RAG
export const evaluateRagQuality = async (evaluationData) => {
  try {
    const token = localStorage.getItem('token');
    if (!token) {
      throw new Error('Требуется авторизация');
    }

    const response = await fetch(`${API_URL}/api/evaluate-rag`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(evaluationData)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Ошибка оценки качества RAG');
    }

    return await response.json();
  } catch (error) {
    console.error('Ошибка при оценке качества RAG:', error);
    throw error;
  }
};

// Удаление документа
export const deleteDocument = async (docId) => {
  try {
    const response = await api.delete(`/documents/${docId}`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при удалении документа');
  }
};

// Очистка истории чата
export const clearChatHistory = async () => {
  try {
    const response = await api.delete('/chats/clear');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при очистке истории чата');
  }
};

// Получение истории запусков оценки RAG
export const getEvaluationHistory = async () => {
  try {
    const response = await api.get('/api/evaluation/history');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении истории запусков оценки');
  }
};

// Скачивание отчёта (json)
export const downloadEvaluationReport = async (filename) => {
  try {
    const response = await api.get(`/api/evaluation/download/${filename}`, { responseType: 'blob' });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при скачивании отчёта');
  }
};

export default api;