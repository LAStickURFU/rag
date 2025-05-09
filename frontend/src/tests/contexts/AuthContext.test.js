import React from 'react';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';

// Отключаем глобальный мок для этого файла
jest.unmock('../../contexts/AuthContext');

// Импортируем после unmock
import { AuthProvider, useAuth } from '../../contexts/AuthContext';

// Мокаем API сервисы
jest.mock('../../services/api');

import * as api from '../../services/api';

// Создаем мок для localStorage
const localStorageMock = (() => {
  let store = {};

  return {
    getItem: jest.fn().mockImplementation((key) => store[key] || null),
    setItem: jest.fn().mockImplementation((key, value) => {
      store[key] = value;
    }),
    removeItem: jest.fn().mockImplementation((key) => {
      delete store[key];
    }),
    clear: jest.fn().mockImplementation(() => {
      store = {};
    }),
    getStore: () => store
  };
})();

// Заменяем глобальный localStorage на наш мок перед запуском всех тестов
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true
});

// Тестовый компонент для использования контекста
const TestComponent = () => {
  const { user, login: authLogin, logout: authLogout, register: authRegister, error, isAuthenticated } = useAuth();
 
  const handleLogin = () => {
    authLogin('testuser', 'password').catch(() => {
      // Обрабатываем ошибку, но ничего не делаем (она уже будет в контексте)
    });
  };
 
  return (
    <div>
      <div data-testid="auth-status">{isAuthenticated ? 'Авторизован' : 'Не авторизован'}</div>
      {error && <div data-testid="auth-error">{error}</div>}
      <button onClick={handleLogin}>Войти</button>
      <button onClick={authLogout}>Выйти</button>
      <button onClick={() => authRegister('newuser', 'password123')}>Зарегистрироваться</button>
    </div>
  );
};

describe('AuthContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
    localStorageMock.getItem.mockClear();
    localStorageMock.setItem.mockClear();
    localStorageMock.removeItem.mockClear();
  });
 
  test('по умолчанию пользователь не авторизован', async () => {
    // Мокируем isAuthenticated, чтобы вернуть false
    api.isAuthenticated.mockReturnValue(false);
   
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
   
    // Ожидаем проверку авторизации
    await waitFor(() => {
      const statusElement = screen.getByTestId('auth-status');
      expect(statusElement).toHaveTextContent('Не авторизован');
    });
  });
 
  test('авторизует пользователя при успешном входе', async () => {
    // Мокируем isAuthenticated и getToken
    api.isAuthenticated.mockReturnValue(false);
    api.getToken.mockImplementation(() => {
      return Promise.resolve({ access_token: 'test-token' });
    });
   
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
   
    // После успешного входа isAuthenticated должен вернуть true
    api.isAuthenticated.mockReturnValue(true);
   
    // Нажимаем кнопку входа
    const loginButton = screen.getByText('Войти');
    await act(async () => {
      fireEvent.click(loginButton);
    });
   
    // Проверяем, что вызван API getToken
    expect(api.getToken).toHaveBeenCalledWith('testuser', 'password');
   
    // Проверяем, что состояние авторизации изменилось
    await waitFor(() => {
      const statusElement = screen.getByTestId('auth-status');
      expect(statusElement).toHaveTextContent('Авторизован');
    });
  });
 
  test('выполняет выход пользователя', async () => {
    // Устанавливаем начальное состояние авторизации
    api.isAuthenticated.mockReturnValue(true);
   
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
   
    // Проверяем, что пользователь считается авторизованным
    await waitFor(() => {
      const statusElement = screen.getByTestId('auth-status');
      expect(statusElement).toHaveTextContent('Авторизован');
    });
   
    // После выхода isAuthenticated должен вернуть false
    api.isAuthenticated.mockReturnValue(false);
   
    // Теперь выходим
    const logoutButton = screen.getByText('Выйти');
    await act(async () => {
      fireEvent.click(logoutButton);
    });
   
    // Проверяем, что вызван API выхода
    expect(api.logout).toHaveBeenCalled();
   
    // Проверяем, что состояние авторизации изменилось
    await waitFor(() => {
      const statusElement = screen.getByTestId('auth-status');
      expect(statusElement).toHaveTextContent('Не авторизован');
    });
  });
});