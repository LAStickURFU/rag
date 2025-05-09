import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../../contexts/AuthContext';
import Header from '../../components/Header';

// Мок для AuthContext с авторизованным пользователем
jest.mock('../../contexts/AuthContext', () => {
  const logoutMock = jest.fn();
  return {
    ...jest.requireActual('../../contexts/AuthContext'),
    useAuth: () => ({
      isAuthenticated: true,
      logout: logoutMock,
    }),
  };
});

describe('Header Component (авторизованный)', () => {
  test('показывает кнопки чата и документов для авторизованных пользователей', () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <Header />
        </AuthProvider>
      </BrowserRouter>
    );
   
    expect(screen.getByText('Чат')).toBeInTheDocument();
    expect(screen.getByText('Документы')).toBeInTheDocument();
    expect(screen.getByText('Выйти')).toBeInTheDocument();
  });

  test('вызывает функцию logout при нажатии на кнопку выхода', () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <Header />
        </AuthProvider>
      </BrowserRouter>
    );
   
    const logoutButton = screen.getByText('Выйти');
    fireEvent.click(logoutButton);
   
    // Проверяем, что функция logout была вызвана
    const { useAuth } = require('../../contexts/AuthContext');
    expect(useAuth().logout).toHaveBeenCalledTimes(1);
  });
});