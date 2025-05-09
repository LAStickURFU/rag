import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import RelevantChunks from '../components/RelevantChunks';

// Мок-данные для тестов
const mockChunks = [
  {
    text: "This is the first chunk of text from the document.",
    relevance: 0.2,
    doc_id: "doc1",
    chunk_id: 1,
    metadata: { title: "Test Document 1" }
  },
  {
    text: "This is the second chunk with more detailed information.",
    relevance: 0.5,
    doc_id: "doc1",
    chunk_id: 2,
    metadata: { title: "Test Document 1" }
  },
  {
    text: "This chunk is from a different document and less relevant.",
    relevance: 0.7,
    doc_id: "doc2",
    chunk_id: 1,
    metadata: { title: "Test Document 2" }
  }
];

describe('RelevantChunks Component', () => {
  test('рендерит сообщение об отсутствии чанков', () => {
    render(<RelevantChunks chunks={[]} />);
    expect(screen.getByText(/Система не использовала никаких документов/i)).toBeInTheDocument();
  });

  test('рендерит все чанки и их тексты', () => {
    render(<RelevantChunks chunks={mockChunks} />);
    // Проверяем, что все чанки отображаются
    expect(screen.getAllByTestId('message-chunk')).toHaveLength(3);
    expect(screen.getByText('This is the first chunk of text from the document.')).toBeInTheDocument();
    expect(screen.getByText('This is the second chunk with more detailed information.')).toBeInTheDocument();
    expect(screen.getByText('This chunk is from a different document and less relevant.')).toBeInTheDocument();
  });

  test('рендерит заголовки документов', () => {
    render(<RelevantChunks chunks={mockChunks} />);
    expect(screen.getAllByText('Test Document 1')).toHaveLength(2);
    expect(screen.getByText('Test Document 2')).toBeInTheDocument();
  });

  test('рендерит бейджи релевантности', () => {
    render(<RelevantChunks chunks={mockChunks} />);
    expect(screen.getByText('Релевантность: 0.80')).toBeInTheDocument();
    expect(screen.getByText('Релевантность: 0.50')).toBeInTheDocument();
    expect(screen.getByText('Релевантность: 0.30')).toBeInTheDocument();
  });
});