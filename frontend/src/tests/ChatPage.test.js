import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatPage from '../pages/ChatPage';
import * as api from '../services/api';

// Мокаем зависимости
jest.mock('../services/api');
jest.mock('../components/ChatInput', () => {
  return function MockChatInput({ onSendMessage }) {
    return (
      <div data-testid="chat-input">
        <button
          onClick={() => onSendMessage('Test message')}
          data-testid="send-button"
        >
          Send
        </button>
      </div>
    );
  };
});

jest.mock('../components/ModelSettings', () => {
  return function MockModelSettings() {
    return <div data-testid="model-settings">Model Settings</div>;
  };
});

jest.mock('../components/RelevantChunks', () => {
  return function MockRelevantChunks({ chunks }) {
    return <div data-testid="message-chunks">Chunks: {chunks?.length || 0}</div>;
  };
});

describe('ChatPage Component', () => {
  beforeEach(() => {
    // Очищаем моки перед каждым тестом
    jest.clearAllMocks();
   
    // Мокаем getChatHistory
    api.getChatHistory.mockResolvedValue([
      {
        id: 1,
        question: "Previous question",
        response: "Previous response",
        created_at: "2023-08-10T12:00:00Z",
        relevant_chunks: [
          {
            text: "Relevant chunk for previous question",
            relevance: 0.2,
            doc_id: "doc1",
            chunk_id: 1
          }
        ]
      }
    ]);
  });

  test('loads chat history with relevant chunks', async () => {
    render(<ChatPage />);
   
    // Дожидаемся загрузки истории чата
    await waitFor(() => {
      expect(screen.getByText('Previous question')).toBeInTheDocument();
      expect(screen.getByText('Previous response')).toBeInTheDocument();
     
      // Проверяем, что чанки отображаются
      expect(screen.getByTestId('message-chunks')).toBeInTheDocument();
    });
  });

  test('sends message and displays response with relevant chunks', async () => {
    // Мокаем функцию askQuestion, возвращаем ответ с чанками
    api.askQuestion.mockResolvedValue({
      id: 2,
      question: "Test message",
      response: "Test response",
      created_at: "2023-08-10T12:01:00Z",
      relevant_chunks: [
        {
          text: "Relevant chunk 1",
          relevance: 0.2,
          doc_id: "doc1",
          chunk_id: 1
        },
        {
          text: "Relevant chunk 2",
          relevance: 0.5,
          doc_id: "doc1",
          chunk_id: 2
        }
      ]
    });
   
    render(<ChatPage />);
   
    // Дожидаемся загрузки истории
    await waitFor(() => {
      expect(screen.getByText('Previous question')).toBeInTheDocument();
    });
   
    // Отправляем сообщение
    act(() => {
      screen.getByTestId('send-button').click();
    });
   
    // Проверяем, что сообщение пользователя отображается
    await waitFor(() => {
      expect(screen.getAllByTestId('user-message').length).toBe(2);
      expect(screen.getByText('Test message')).toBeInTheDocument();
    });
   
    // Проверяем, что ответ бота и чанки отображаются
    await waitFor(() => {
      expect(screen.getAllByTestId('bot-message').length).toBe(2);
      expect(screen.getByText('Test response')).toBeInTheDocument();
     
      // Проверяем отображение чанков (должен быть один блок с чанками)
      const chunkBlocks = screen.getAllByTestId('message-chunks');
      expect(chunkBlocks.length).toBe(1);
      // Проверяем, что в блоке отображается правильное количество чанков
      expect(chunkBlocks[0]).toHaveTextContent('Chunks: 2');
    });
  });
});