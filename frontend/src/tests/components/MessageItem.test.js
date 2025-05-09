import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import '@testing-library/jest-dom/extend-expect';
import MessageItem from '../../components/MessageItem';

describe('MessageItem Component', () => {
  test('рендерит сообщение пользователя корректно', () => {
    const message = 'Это сообщение пользователя';
    render(<MessageItem message={message} isUser={true} />);
   
    const messageElement = screen.getByText(message);
    expect(messageElement).toBeInTheDocument();
   
    // Проверяем наличие иконки пользователя (PersonIcon)
    const personIcon = screen.getByTestId('PersonIcon');
    expect(personIcon).toBeInTheDocument();
  });

  test('рендерит сообщение системы корректно', () => {
    const message = 'Это ответное сообщение системы';
    render(<MessageItem message={message} isUser={false} />);
   
    const messageElement = screen.getByText(message);
    expect(messageElement).toBeInTheDocument();
   
    // Проверяем наличие иконки робота (SmartToyIcon)
    const robotIcon = screen.getByTestId('SmartToyIcon');
    expect(robotIcon).toBeInTheDocument();
  });

  test('применяет различные стили для сообщений пользователя и системы', () => {
    // Рендерим сообщение пользователя
    const { container: userContainer } = render(
      <MessageItem message="Сообщение пользователя" isUser={true} />
    );
   
    // Получаем элемент сообщения пользователя
    const userMessageElement = userContainer.querySelector('.MuiPaper-root');
   
    // Рендерим сообщение системы
    const { container: systemContainer } = render(
      <MessageItem message="Сообщение системы" isUser={false} />
    );
   
    // Получаем элемент сообщения системы
    const systemMessageElement = systemContainer.querySelector('.MuiPaper-root');
   
    // Проверяем, что стили различаются (цвет фона)
    // Вместо проверки на наличие конкретных классов, которые могут меняться,
    // проверяем, что стили сообщений пользователя и системы отличаются
    expect(userMessageElement.className).not.toBe(systemMessageElement.className);
  });
});