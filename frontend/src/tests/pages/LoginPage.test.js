import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import '@testing-library/jest-dom/extend-expect';
import { BrowserRouter } from 'react-router-dom';
import LoginPage from '../../pages/LoginPage';

// Мокаем useNavigate, чтобы не было реальных переходов
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
}));

// Мокаем контекст авторизации
const loginMock = jest.fn();
const mockAuthContext = {
  login: loginMock,
      loading: false,
      error: null
  };

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => mockAuthContext
}));

describe('LoginPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Для каждого теста устанавливаем loading в false
    mockAuthContext.loading = false;
    mockAuthContext.error = null;
  });
 
  test('рендерит форму входа', () => {
    render(
      <BrowserRouter>
          <LoginPage />
      </BrowserRouter>
    );
    expect(screen.getByText('RAG Сервис')).toBeInTheDocument();
    expect(screen.getByText('Вход')).toBeInTheDocument();
    expect(screen.getByText('Регистрация')).toBeInTheDocument();
    expect(screen.getByTestId('username-input')).toBeInTheDocument();
    expect(screen.getByTestId('password-input')).toBeInTheDocument();
    expect(screen.getByTestId('login-button')).toBeInTheDocument();
  });
 
  test('обновляет поля ввода при вводе данных', () => {
    render(
      <BrowserRouter>
          <LoginPage />
      </BrowserRouter>
    );
    const usernameInput = screen.getByTestId('username-input').querySelector('input');
    const passwordInput = screen.getByTestId('password-input').querySelector('input');
    fireEvent.change(usernameInput, { target: { value: 'testuser' } });
    fireEvent.change(passwordInput, { target: { value: 'password123' } });
    expect(usernameInput.value).toBe('testuser');
    expect(passwordInput.value).toBe('password123');
  });
 
  test('показывает ошибку при отправке формы с пустыми полями', async () => {
    render(
      <BrowserRouter>
          <LoginPage />
      </BrowserRouter>
    );
    const loginButton = screen.getByTestId('login-button');
    fireEvent.click(loginButton);
    await waitFor(() => {
      expect(screen.getByText('Заполните все поля')).toBeInTheDocument();
    });
  });
 
  test('рендерит табы для переключения между входом и регистрацией', () => {
    render(
      <BrowserRouter>
          <LoginPage />
      </BrowserRouter>
    );
    expect(screen.getByText('Вход')).toBeInTheDocument();
    expect(screen.getByText('Регистрация')).toBeInTheDocument();
  });
 
  test('отображает индикатор загрузки при loading=true', () => {
    // Обёртка для управления loading
    function LoginPageWithLoading() {
      const [loading, setLoading] = React.useState(true);
      return <LoginPage loading={loading} setLoading={setLoading} />;
    }
    render(
      <BrowserRouter>
        <LoginPageWithLoading />
      </BrowserRouter>
    );
    expect(screen.getByTestId('login-button')).toBeDisabled();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    React.useState.mockRestore && React.useState.mockRestore();
  });
});