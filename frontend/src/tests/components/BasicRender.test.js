import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BrowserRouter } from 'react-router-dom';
import ChatInput from '../../components/ChatInput';
import MessageItem from '../../components/MessageItem';
import Header from '../../components/Header';
import DocumentList from '../../components/DocumentList';

// Мок для AuthContext
jest.mock('../../contexts/AuthContext', () => ({
  ...jest.requireActual('../../contexts/AuthContext'),
  useAuth: () => ({
    isAuthenticated: false,
    logout: jest.fn(),
  }),
}));

// Мок для scrollIntoView
Element.prototype.scrollIntoView = jest.fn();

describe('Basic Rendering Tests', () => {
  describe('ChatInput', () => {
    test('рендерит компонент ChatInput', () => {
      const mockSendMessage = jest.fn();
      render(<ChatInput onSendMessage={mockSendMessage} loading={false} />);
     
      expect(screen.getByPlaceholderText('Введите ваш вопрос...')).toBeInTheDocument();
      expect(screen.getByText('Отправить')).toBeInTheDocument();
    });

    test('отключает ввод при loading=true', () => {
      const mockSendMessage = jest.fn();
      render(<ChatInput onSendMessage={mockSendMessage} loading={true} />);
     
      expect(screen.getByPlaceholderText('Введите ваш вопрос...')).toBeDisabled();
      expect(screen.getByText('Отправка...')).toBeDisabled();
    });
  });

  describe('MessageItem', () => {
    test('рендерит сообщение пользователя', () => {
      render(<MessageItem message="Тестовое сообщение пользователя" isUser={true} />);
     
      expect(screen.getByText('Тестовое сообщение пользователя')).toBeInTheDocument();
    });

    test('рендерит сообщение системы', () => {
      render(<MessageItem message="Тестовое сообщение системы" isUser={false} />);
     
      expect(screen.getByText('Тестовое сообщение системы')).toBeInTheDocument();
    });
  });

  describe('Header', () => {
    test('рендерит заголовок приложения', () => {
      render(
        <BrowserRouter>
          <Header />
        </BrowserRouter>
      );
     
      expect(screen.getByText('RAG-сервис')).toBeInTheDocument();
    });
  });

  describe('DocumentList', () => {
    test('показывает сообщение при пустом списке документов', () => {
      render(<DocumentList documents={[]} loading={false} />);
     
      expect(screen.getByText('Загруженные документы')).toBeInTheDocument();
      expect(screen.getByText('У вас пока нет загруженных документов.')).toBeInTheDocument();
    });

    test('показывает индикатор загрузки', () => {
      render(<DocumentList documents={[]} loading={true} />);
     
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });
});