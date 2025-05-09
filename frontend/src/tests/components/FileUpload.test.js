import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import FileUpload from '../../components/FileUpload';

describe('FileUpload Component', () => {
  const mockUpload = jest.fn();
 
  beforeEach(() => {
    jest.clearAllMocks();
  });
 
  test('рендерит форму загрузки файла', () => {
    render(<FileUpload onUpload={mockUpload} loading={false} />);
    expect(screen.getByText('Загрузить новые документы')).toBeInTheDocument();
    expect(screen.getByText('Выбрать файлы')).toBeInTheDocument();
    expect(screen.getByTestId('submit-button')).toBeInTheDocument();
    // Эмулируем загрузку файла, чтобы появились поля title/source
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByTestId('file-input');
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    expect(screen.getByTestId('title-input')).toBeInTheDocument();
    expect(screen.getByTestId('source-select')).toBeInTheDocument();
  });
 
  test('обновляет поле названия документа при вводе', () => {
    render(<FileUpload onUpload={mockUpload} loading={false} />);
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByTestId('file-input');
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    const titleInput = screen.getByTestId('title-input');
    fireEvent.change(titleInput, { target: { value: 'Тестовый документ' } });
    expect(titleInput.value).toBe('Тестовый документ');
  });
 
  test('кнопка отправки по умолчанию отключена', () => {
    render(<FileUpload onUpload={mockUpload} loading={false} />);
   
    const submitButton = screen.getByTestId('submit-button');
    expect(submitButton).toBeDisabled();
  });
 
  test('отключает элементы формы при loading=true', () => {
    render(<FileUpload onUpload={mockUpload} loading={true} />);
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByTestId('file-input');
    act(() => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    const titleInput = screen.getByTestId('title-input');
    expect(titleInput).toBeDisabled();
   
    // Находим элементы с помощью других атрибутов
    const submitButton = screen.getByTestId('submit-button');
    expect(submitButton).toBeDisabled();
    expect(submitButton).toHaveTextContent('Загрузка...');
   
    // Проверяем что кнопка выбора файла недоступна
    const fileButton = screen.getByText('Выбрать файлы').closest('label');
    expect(fileButton).toHaveAttribute('aria-disabled', 'true');
  });
 
  test('вызывает onUpload с правильными параметрами при отправке формы', async () => {
    render(<FileUpload onUpload={mockUpload} loading={false} />);
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByTestId('file-input');
    await act(async () => {
      fireEvent.change(fileInput, { target: { files: [file] } });
    });
    const titleInput = screen.getByTestId('title-input');
    fireEvent.change(titleInput, { target: { value: 'Тестовый документ' } });
    const submitButton = screen.getByTestId('submit-button');
    expect(submitButton).not.toBeDisabled();
    mockUpload.mockResolvedValue(true);
    await act(async () => {
      fireEvent.click(submitButton);
    });
    expect(mockUpload).toHaveBeenCalled();
  });
});