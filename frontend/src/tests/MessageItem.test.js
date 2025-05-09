import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MessageItem from '../components/MessageItem';

// Мокаем компонент RelevantChunks
jest.mock('../components/RelevantChunks', () => {
  return function MockRelevantChunks({ chunks }) {
    return chunks && chunks.length > 0 ? (
      <div data-testid="relevant-chunks">
        <span>Найдено чанков: {chunks.length}</span>
      </div>
    ) : null;
  };
});

describe('MessageItem Component', () => {
  test('renders user message correctly', () => {
    render(<MessageItem message="Hello, this is a user message" isUser={true} />);
   
    // Проверяем, что сообщение отображается
    expect(screen.getByText('Hello, this is a user message')).toBeInTheDocument();
   
    // У сообщения пользователя не должно быть релевантных чанков
    expect(screen.queryByTestId('relevant-chunks')).not.toBeInTheDocument();
  });

  test('renders bot message without chunks correctly', () => {
    render(<MessageItem message="This is a response from the bot" isUser={false} />);
   
    // Проверяем, что сообщение отображается
    expect(screen.getByText('This is a response from the bot')).toBeInTheDocument();
   
    // Без указания чанков компонент не должен их отображать
    expect(screen.queryByTestId('relevant-chunks')).not.toBeInTheDocument();
  });

  test('renders bot message with chunks correctly', () => {
    const mockChunks = [
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
    ];
   
    render(
      <MessageItem
        message="This is a response from the bot with relevant chunks"
        isUser={false}
        relevantChunks={mockChunks}
      />
    );
   
    // Проверяем, что сообщение отображается
    expect(screen.getByText('This is a response from the bot with relevant chunks')).toBeInTheDocument();
   
    // Проверяем, что отображаются релевантные чанки
    expect(screen.getByTestId('relevant-chunks')).toBeInTheDocument();
    expect(screen.getByText('Найдено чанков: 2')).toBeInTheDocument();
  });

  test('does not render chunks for user messages', () => {
    const mockChunks = [
      {
        text: "Relevant chunk 1",
        relevance: 0.2,
        doc_id: "doc1",
        chunk_id: 1
      }
    ];
   
    render(
      <MessageItem
        message="User message with chunks that should be ignored"
        isUser={true}
        relevantChunks={mockChunks}
      />
    );
   
    // Проверяем, что сообщение отображается
    expect(screen.getByText('User message with chunks that should be ignored')).toBeInTheDocument();
   
    // Для сообщений пользователя не должны отображаться чанки, даже если они переданы
    expect(screen.queryByTestId('relevant-chunks')).not.toBeInTheDocument();
  });

  test('handles empty chunks array for bot messages', () => {
    render(
      <MessageItem
        message="Bot message with empty chunks array"
        isUser={false}
        relevantChunks={[]}
      />
    );
   
    // Проверяем, что сообщение отображается
    expect(screen.getByText('Bot message with empty chunks array')).toBeInTheDocument();
   
    // С пустым массивом чанков компонент не должен их отображать
    expect(screen.queryByTestId('relevant-chunks')).not.toBeInTheDocument();
  });
});