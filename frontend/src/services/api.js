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
    files.forEach((file, index) => {
      // Явно указываем имя файла и другие его свойства
      formData.append('files', file, file.name);
    });
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

// Получение информации о текущем пользователе
export const getCurrentUser = async () => {
  try {
    const response = await api.get('/me');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении данных пользователя');
  }
};

// Обновление роли пользователя (только для админов)
export const updateUserRole = async (username, role) => {
  try {
    const response = await api.post(`/users/${username}/role?role=${role}`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при обновлении роли пользователя');
  }
};

// Изменение пароля текущего пользователя
export const changePassword = async (currentPassword, newPassword) => {
  try {
    const response = await api.post('/change-password', {
      current_password: currentPassword,
      new_password: newPassword
    });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при изменении пароля');
  }
};

// Сброс пароля пользователя (только для админов)
export const resetUserPassword = async (username, newPassword) => {
  try {
    const response = await api.post(`/users/${username}/reset-password?new_password=${newPassword}`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при сбросе пароля пользователя');
  }
};

// Получение списка документов
export const getDocuments = async (allUsers = false, page = 0, pageSize = 100, returnAll = false) => {
  try {
    let url = '/documents?';
    const params = [];
    
    if (allUsers) params.push('all_users=true');
    if (page > 0) params.push(`page=${page}`);
    if (pageSize !== 100) params.push(`page_size=${pageSize}`);
    if (returnAll) params.push('return_all=true');
    
    url += params.join('&');
    
    const response = await api.get(url);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении списка документов');
  }
};

// Переиндексация документов пользователя или всех документов
export const reindexDocuments = async (allUsers = false) => {
  try {
    const response = await api.post(`/documents/reindex${allUsers ? '?all_users=true' : ''}`);
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

// Очистка индекса векторной базы
export const clearRagIndex = async (userId = null) => {
  try {
    const url = '/index/clear' + (userId ? `?user_id=${userId}` : '');
    const response = await api.post(url);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при очистке индекса');
  }
};

// Удаление всех документов (только для админов)
export const deleteAllDocuments = async () => {
  try {
    const response = await api.delete('/documents/all/clear');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при удалении всех документов');
  }
};

// Получение списка всех пользователей (только для админов)
export const getAllUsers = async () => {
  try {
    const response = await api.get('/users');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении списка пользователей');
  }
};

// Удаление пользователя (только для админов)
export const deleteUser = async (username) => {
  try {
    const response = await api.delete(`/users/${username}`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при удалении пользователя');
  }
};

// Получение списка доступных моделей
export const getAvailableModels = async () => {
  try {
    const response = await api.get('/model/available');
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при получении списка моделей');
  }
};

// Переключение на другую модель
export const switchModel = async (modelName) => {
  try {
    const response = await api.post(`/model/switch/${modelName}`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : new Error('Ошибка при переключении модели');
  }
};

export default api;