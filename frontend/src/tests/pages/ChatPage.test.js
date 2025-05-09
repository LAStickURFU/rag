import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatPage from '../../pages/ChatPage';

// Мокаем API сервисы
jest.mock('../../services/api', () => ({
  askQuestion: jest.fn(),
  getChatHistory: jest.fn()
}));

import { askQuestion, getChatHistory } from '../../services/api';

describe('ChatPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Устанавливаем моки по умолчанию
    getChatHistory.mockResolvedValue([]);
  });
 
  test('рендерит пустую страницу чата', async () => {
    await act(async () => {
      render(<ChatPage />);
    });
   
    // Ожидаем загрузку истории
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
   
    expect(screen.getByText('Чат с RAG-системой')).toBeInTheDocument();
    expect(screen.getByText('Задайте вопрос, чтобы начать общение')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Введите ваш вопрос...')).toBeInTheDocument();
    expect(screen.getByText('Отправить')).toBeInTheDocument();
  });
 
  test('показывает индикатор загрузки при инициализации', async () => {
    // Создаем промис, который никогда не разрешается
    getChatHistory.mockImplementation(() => new Promise(() => {})); // Бесконечная загрузка
   
    await act(async () => {
      render(<ChatPage />);
    });
   
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
 
  test('загружает и отображает историю чата', async () => {
    const mockChatHistory = [
      { id: 1, question: 'Вопрос 1', response: 'Ответ 1' },
      { id: 2, question: 'Вопрос 2', response: 'Ответ 2' }
    ];
   
    getChatHistory.mockResolvedValue(mockChatHistory);
   
    await act(async () => {
      render(<ChatPage />);
    });
   
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
   
    // Дожидаемся появления тестов на экране
    await waitFor(() => {
    expect(screen.getByText('Вопрос 1')).toBeInTheDocument();
    expect(screen.getByText('Ответ 1')).toBeInTheDocument();
    expect(screen.getByText('Вопрос 2')).toBeInTheDocument();
    expect(screen.getByText('Ответ 2')).toBeInTheDocument();
    });
  });
 
  test('отправляет сообщение и отображает ответ', async () => {
    // Мокаем запрос к API
    askQuestion.mockResolvedValue({ response: 'Ответ на тестовый вопрос' });
    getChatHistory.mockResolvedValue([]);
   
    await act(async () => {
      render(<ChatPage />);
    });
   
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
   
    // Вводим и отправляем сообщение
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
   
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
    });
   
    const button = screen.getByText('Отправить');
    await act(async () => {
      fireEvent.click(button);
    });
   
    // Проверяем, что вопрос отображается
    expect(screen.getByText('Тестовый вопрос')).toBeInTheDocument();
   
    // Проверяем вызов API
    expect(askQuestion).toHaveBeenCalledWith('Тестовый вопрос');
   
    // Ожидаем ответ
    await waitFor(() => {
      expect(screen.getByText('Ответ на тестовый вопрос')).toBeInTheDocument();
    });
  });
 
  test('показывает ошибку, если запрос не выполнен', async () => {
    // Мокаем ошибку API
    askQuestion.mockRejectedValue(new Error('Ошибка API'));
    getChatHistory.mockResolvedValue([]);
   
    await act(async () => {
      render(<ChatPage />);
    });
   
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
   
    // Вводим и отправляем сообщение
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
   
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
    });
   
    const button = screen.getByText('Отправить');
    await act(async () => {
      fireEvent.click(button);
    });
   
    // Проверяем отображение ошибки
    await waitFor(() => {
      expect(screen.getByText('Не удалось получить ответ')).toBeInTheDocument();
    });
  });
 
  test('отключает ввод при отправке сообщения', async () => {
    // Имитируем задержку ответа
    askQuestion.mockImplementation(() => new Promise(resolve => {
      setTimeout(() => resolve({ response: 'Ответ' }), 100);
    }));
    getChatHistory.mockResolvedValue([]);
   
    await act(async () => {
      render(<ChatPage />);
    });
   
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
   
    // Вводим и отправляем сообщение
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
   
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
    });
   
    const button = screen.getByText('Отправить');
    await act(async () => {
      fireEvent.click(button);
    });
   
    // Проверяем, что кнопка отключена во время загрузки
    expect(screen.getByText('Отправка...')).toBeDisabled();
    expect(input).toBeDisabled();
   
    // Ожидаем завершения запроса
    await waitFor(() => {
      expect(screen.getByText('Ответ')).toBeInTheDocument();
    }, { timeout: 200 });
   
    // Проверяем, что интерфейс снова активен
    expect(screen.getByText('Отправить')).toBeInTheDocument();
    expect(input).not.toBeDisabled();
  });
 
  test('отображает релевантные чанки при получении ответа', async () => {
    const mockChunks = [
      { text: 'Чанк 1', relevance: 0.1, metadata: { title: 'Документ 1', source: 'manual' } },
      { text: 'Чанк 2', relevance: 0.2, metadata: { title: 'Документ 2', source: 'import' } }
    ];
    askQuestion.mockResolvedValue({ response: 'Ответ', relevant_chunks: mockChunks });
    getChatHistory.mockResolvedValue([]);
    await act(async () => {
      render(<ChatPage />);
    });
    await waitFor(() => {
      expect(getChatHistory).toHaveBeenCalled();
    });
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
    await act(async () => {
      fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
    });
    const button = screen.getByText('Отправить');
    await act(async () => {
      fireEvent.click(button);
    });
    await waitFor(() => {
      expect(screen.getAllByTestId('message-chunk').length).toBe(2);
      expect(screen.getByText('Чанк 1')).toBeInTheDocument();
      expect(screen.getByText('Чанк 2')).toBeInTheDocument();
    });
  });
});