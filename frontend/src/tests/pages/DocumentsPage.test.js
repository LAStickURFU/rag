import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import DocumentsPage from '../../pages/DocumentsPage';

import * as api from '../../services/api';
jest.mock('../../services/api', () => ({
  ...jest.requireActual('../../services/api'),
  uploadDocuments: jest.fn(() => Promise.resolve()),
  getDocuments: jest.fn(() => Promise.resolve([])),
  uploadDocument: jest.fn(() => Promise.resolve()),
}));

describe('DocumentsPage', () => {
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
    }
  ];
 
  beforeEach(() => {
    jest.clearAllMocks();
    // Мок для преобразования даты
    const originalToLocaleDateString = Date.prototype.toLocaleDateString;
    Date.prototype.toLocaleDateString = jest.fn().mockImplementation(() => 'Форматированная дата');
   
    // Устанавливаем моки по умолчанию
    api.getDocuments.mockResolvedValue(mockDocuments);
  });
 
  test('рендерит страницу документов', async () => {
    await act(async () => {
      render(<DocumentsPage />);
    });
   
    // Проверяем заголовок
    const heading = screen.getByText('Управление документами');
    expect(heading).toBeInTheDocument();
   
    // Ожидаем загрузку документов
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled();
    });
   
    // Проверяем компоненты на странице
    const uploadSectionTitle = screen.getByText('Загрузить новые документы');
    expect(uploadSectionTitle).toBeInTheDocument();
   
    const documentListTitle = screen.getByText('Загруженные документы');
    expect(documentListTitle).toBeInTheDocument();
   
    // Проверяем наличие формы загрузки
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]');
    act(() => {
      Object.defineProperty(fileInput, 'files', { value: [file] });
      fireEvent.change(fileInput);
    });
    const titleInput = screen.getAllByTestId('title-input')[0];
    expect(titleInput).toBeInTheDocument();
   
    const uploadButton = screen.getByTestId('submit-button');
  });
 
  test('отображает список загруженных документов', async () => {
    await act(async () => {
      render(<DocumentsPage />);
    });
   
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled();
    });
   
    // Проверяем, что документы отображаются
    const doc1 = screen.getByText('Документ 1');
    expect(doc1).toBeInTheDocument();
   
    const doc2 = screen.getByText('Документ 2');
    expect(doc2).toBeInTheDocument();
   
    const manualSource = screen.getByText('manual');
    expect(manualSource).toBeInTheDocument();
   
    const webSource = screen.getByText('web');
    expect(webSource).toBeInTheDocument();
  });
 
  test('показывает ошибку, если не удалось загрузить документы', async () => {
    // Мокаем ошибку API
    api.getDocuments.mockRejectedValue(new Error('Ошибка загрузки документов'));
   
    await act(async () => {
      render(<DocumentsPage />);
    });
   
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled();
    });
   
    // Проверяем отображение ошибки
    const errorMessage = screen.getByText('Не удалось загрузить список документов');
    expect(errorMessage).toBeInTheDocument();
  });
 
  test('загружает документ при отправке формы', async () => {
    // Мок для успешной загрузки
    api.uploadDocuments.mockResolvedValue([{ id: 3, title: 'Новый документ' }]);
   
    await act(async () => {
      render(<DocumentsPage />);
    });
   
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled();
    });
   
    // Заполняем форму
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]');
    act(() => {
      Object.defineProperty(fileInput, 'files', { value: [file] });
      fireEvent.change(fileInput);
    });
    const titleInput = screen.getAllByTestId('title-input')[0];
    expect(titleInput).toBeInTheDocument();
   
    // Отправляем форму
    mockDocuments.push({ id: 3, title: 'Новый документ', source: 'manual', created_at: new Date().toISOString() });
    api.getDocuments.mockResolvedValueOnce(mockDocuments);
   
    const submitButton = screen.getByText('Загрузить документы');
    await act(async () => {
      fireEvent.click(submitButton);
    });
   
    // Проверяем вызов API
    await waitFor(() => {
      expect(api.uploadDocuments).toHaveBeenCalled();
    });
   
    // Проверяем, что список документов обновляется
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalledTimes(2); // Первый раз при загрузке, второй раз после загрузки документа
    });
   
    // Проверяем сообщение об успехе
    const successMessage = screen.getByText(text => text.includes('успешно'));
    expect(successMessage).toBeInTheDocument();
  });
 
  test('показывает ошибку при неудачной загрузке документа', async () => {
    // Мок для неудачной загрузки
    api.uploadDocuments.mockRejectedValue(new Error('Ошибка загрузки документа'));
   
    await act(async () => {
      render(<DocumentsPage />);
    });
   
    await waitFor(() => {
      expect(api.getDocuments).toHaveBeenCalled();
    });
   
    // Заполняем форму
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = document.querySelector('input[type="file"]');
    act(() => {
      Object.defineProperty(fileInput, 'files', { value: [file] });
      fireEvent.change(fileInput);
    });
    const titleInput = screen.getAllByTestId('title-input')[0];
    await act(async () => {
      fireEvent.change(titleInput, { target: { value: 'Новый документ' } });
    });
   
    // Отправляем форму
    const submitButton = screen.getByText('Загрузить документы');
    await act(async () => {
      fireEvent.click(submitButton);
    });
   
    // Проверяем отображение ошибки
    await waitFor(() => {
      const errorMessages = screen.getAllByText((text) => text.includes('Ошибка загрузки документа'));
      expect(errorMessages.length).toBeGreaterThan(0);
    });
  });
});