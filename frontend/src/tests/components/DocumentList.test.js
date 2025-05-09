import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import '@testing-library/jest-dom/extend-expect';
import DocumentList from '../../components/DocumentList';

describe('DocumentList Component', () => {
  const mockDocuments = [
    {
      id: 1,
      title: 'Документ 1',
      source: 'manual',
      created_at: '2023-05-10T14:30:00Z'
    },
    {
      id: 2,
      title: 'Документ 2',
      source: 'web',
      created_at: '2023-05-11T10:15:00Z'
    },
    {
      id: 3,
      title: 'Документ 3',
      source: 'pdf',
      created_at: '2023-05-12T08:45:00Z'
    }
  ];
 
  test('показывает индикатор загрузки при loading=true', () => {
    render(<DocumentList documents={[]} loading={true} />);
   
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
  });
 
  test('показывает сообщение, когда документов нет', () => {
    render(<DocumentList documents={[]} loading={false} />);
   
    const title = screen.getByText('Загруженные документы');
    expect(title).toBeInTheDocument();
   
    const emptyMessage = screen.getByText('У вас пока нет загруженных документов.');
    expect(emptyMessage).toBeInTheDocument();
  });
 
  test('рендерит список документов корректно', () => {
    render(<DocumentList documents={mockDocuments} loading={false} />);
   
    const title = screen.getByText('Загруженные документы');
    expect(title).toBeInTheDocument();
   
    // Проверяем наличие всех документов в списке
    const doc1 = screen.getByText('Документ 1');
    expect(doc1).toBeInTheDocument();
   
    const doc2 = screen.getByText('Документ 2');
    expect(doc2).toBeInTheDocument();
   
    const doc3 = screen.getByText('Документ 3');
    expect(doc3).toBeInTheDocument();
  });
 
  test('отображает правильные источники для документов', () => {
    render(<DocumentList documents={mockDocuments} loading={false} />);
   
    // Проверяем отображение чипов с источниками
    const manualSource = screen.getByText('manual');
    expect(manualSource).toBeInTheDocument();
   
    const webSource = screen.getByText('web');
    expect(webSource).toBeInTheDocument();
   
    const pdfSource = screen.getByText('pdf');
    expect(pdfSource).toBeInTheDocument();
  });
 
  test('отображает форматированную дату добавления документов', () => {
    // Мокируем функцию форматирования даты
    const originalDate = global.Date;
    const mockDate = jest.fn(() => ({
      toLocaleDateString: jest.fn().mockReturnValue('Форматированная дата'),
      toLocaleTimeString: jest.fn().mockReturnValue('10:00'),
    }));
   
    mockDate.UTC = originalDate.UTC;
    mockDate.parse = originalDate.parse;
    global.Date = mockDate;
   
    render(<DocumentList documents={mockDocuments} loading={false} />);
   
    // После рендеринга восстанавливаем оригинальный Date
    global.Date = originalDate;
   
    // Проверяем отображение дат
    // Используем менее строгую проверку, поскольку форматирование может различаться
    const dateElements = screen.getAllByText(/Форматированная дата|10:00/i);
    expect(dateElements.length).toBeGreaterThan(0);
  });
});