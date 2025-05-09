import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/extend-expect';
import ChatInput from '../../components/ChatInput';

describe('ChatInput Component', () => {
  const mockSendMessage = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('рендерит поле ввода и кнопку отправки', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={false} />);
   
    const inputElement = screen.getByPlaceholderText('Введите ваш вопрос...');
    expect(inputElement).toBeInTheDocument();
   
    const buttonElement = screen.getByText('Отправить');
    expect(buttonElement).toBeInTheDocument();
  });

  test('обновляет значение ввода при вводе текста', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={false} />);
   
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
    fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
   
    expect(input.value).toBe('Тестовый вопрос');
  });

  test('вызывает функцию onSendMessage при отправке формы', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={false} />);
    const inputElement = screen.getByPlaceholderText('Введите ваш вопрос...');
    fireEvent.change(inputElement, { target: { value: 'Тестовый вопрос' } });
    const button = screen.getByText('Отправить');
    fireEvent.click(button);
    expect(mockSendMessage).toHaveBeenCalledWith('Тестовый вопрос', false);
  });

  test('очищает поле ввода после отправки', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={false} />);
   
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
    fireEvent.change(input, { target: { value: 'Тестовый вопрос' } });
   
    const button = screen.getByText('Отправить');
    fireEvent.click(button);
   
    expect(input.value).toBe('');
  });

  test('отключает кнопку отправки, когда loading=true', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={true} />);
   
    const button = screen.getByText('Отправка...');
    expect(button).toHaveAttribute('disabled');
  });

  test('отключает поле ввода, когда loading=true', () => {
    render(<ChatInput onSendMessage={mockSendMessage} loading={true} />);
   
    const input = screen.getByPlaceholderText('Введите ваш вопрос...');
    expect(input).toHaveAttribute('disabled');
  });
});