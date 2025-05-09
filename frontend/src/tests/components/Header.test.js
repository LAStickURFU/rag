import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../../contexts/AuthContext';
import Header from '../../components/Header';

// Мок для AuthContext
jest.mock('../../contexts/AuthContext', () => ({
  ...jest.requireActual('../../contexts/AuthContext'),
  useAuth: () => ({
    isAuthenticated: false,
    logout: jest.fn(),
  }),
}));

describe('Header Component', () => {
  test('рендерит заголовок приложения', () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <Header />
        </AuthProvider>
      </BrowserRouter>
    );
   
    expect(screen.getByText('RAG-сервис')).toBeInTheDocument();
  });

  test('показывает кнопки входа и регистрации для неавторизованных пользователей', () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <Header />
        </AuthProvider>
      </BrowserRouter>
    );
   
    expect(screen.getByText('Войти')).toBeInTheDocument();
    expect(screen.getByText('Регистрация')).toBeInTheDocument();
  });
});