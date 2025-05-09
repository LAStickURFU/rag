import React from 'react';
import { Box } from '@mui/material';

const DashboardPage = () => (
  <div style={{ padding: 32 }}>
    <h1>Добро пожаловать в RAG Dashboard</h1>
    <Box component="span">Это главная страница вашей Retrieval-Augmented Generation системы.</Box>
    <ul>
      <li>Загрузка документов</li>
      <li>Поиск и генерация ответов</li>
      <li>Настройки и мониторинг</li>
    </ul>
  </div>
);

export default DashboardPage;