import React, { useState, useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import api from '../services/api';

// Компонент для отображения краткой статистики на странице документов
const DocumentStats = ({ documentStats, loading }) => {
  const [stats, setStats] = useState(null);
  const [statError, setStatError] = useState(null);
  const [statLoading, setStatLoading] = useState(true);
  const [lastFetchTime, setLastFetchTime] = useState(0);
  const fetchingRef = useRef(false); // Флаг, чтобы избежать параллельных запросов

  useEffect(() => {
    // Если документы не загружены, пропускаем запрос
    if (!documentStats || !Array.isArray(documentStats) || documentStats.length === 0 || loading) {
      return;
    }
    
    // Ограничиваем частоту запросов статистики (не чаще чем раз в 5 секунд)
    const now = Date.now();
    if (now - lastFetchTime < 5000 && stats) {
      return;
    }
    
    // Предотвращаем параллельные запросы
    if (fetchingRef.current) {
      return;
    }
    
    const fetchStats = async () => {
      if (fetchingRef.current) return;
      try {
        fetchingRef.current = true;
        setStatLoading(true);
        console.log('Запрашиваем статистику системы');
        const response = await api.get('/stats');
        setStats(response.data);
        setStatError(null);
        setLastFetchTime(Date.now());
      } catch (err) {
        console.error('Error fetching stats:', err);
        setStatError('Ошибка при получении статистики');
      } finally {
        setStatLoading(false);
        fetchingRef.current = false;
      }
    };

    fetchStats();
  }, [documentStats, loading, lastFetchTime]); // Удаляем stats из зависимостей

  if (loading || statLoading) {
    return null; // Ничего не показываем во время загрузки
  }

  if (statError) {
    return null; // Ничего не показываем при ошибке
  }

  if (!stats) {
    return null;
  }

  return (
    <Box sx={{ mb: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
      <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'medium' }}>
        Статистика базы знаний:
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        <Box>
          <Typography variant="body2" color="textSecondary">
            Всего документов: <strong>{stats.documents.total}</strong>
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Проиндексировано: <strong>{stats.documents.status_counts.indexed || 0}</strong>
          </Typography>
        </Box>
        <Box>
          <Typography variant="body2" color="textSecondary">
            Всего фрагментов: <strong>{stats.chunks.total_count}</strong>
          </Typography>
          <Typography variant="body2" color="textSecondary">
            В среднем на документ: <strong>{stats.chunks.per_document.avg ? Math.round(stats.chunks.per_document.avg) : 'N/A'}</strong>
          </Typography>
        </Box>
        <Box>
          <Typography variant="body2" color="textSecondary">
            Точек в векторной базе: <strong>{stats.vector_db.points_count || 0}</strong>
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Модель: <strong>{stats.search.model_name ? stats.search.model_name.split('/').pop() : 'N/A'}</strong>
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default DocumentStats; 