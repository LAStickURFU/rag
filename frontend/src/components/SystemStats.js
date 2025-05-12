import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  CircularProgress, 
  Grid, 
  Divider, 
  Chip,
  Card,
  CardContent,
  Alert,
  IconButton,
  Tooltip,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import StorageIcon from '@mui/icons-material/Storage';
import DescriptionIcon from '@mui/icons-material/Description';
import PieChartIcon from '@mui/icons-material/PieChart';
import api from '../services/api';

// Компонент для отображения количества документов по статусам
const DocumentStatusCard = ({ stats }) => {
  if (!stats || !stats.documents) return null;

  const { total, status_counts } = stats.documents;
  
  const statusLabels = {
    'indexed': 'Проиндексировано',
    'error': 'Ошибка',
    'processing': 'Обработка',
    'chunking': 'Разбиение на чанки',
    'embedding': 'Создание эмбеддингов',
    'reindexing': 'Переиндексация',
    'uploaded': 'Загружено'
  };

  const statusColors = {
    'indexed': 'success',
    'error': 'error',
    'processing': 'primary',
    'chunking': 'primary',
    'embedding': 'primary',
    'reindexing': 'secondary',
    'uploaded': 'info'
  };

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          <DescriptionIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Документы ({total})
        </Typography>
        <Divider sx={{ my: 1 }} />
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, my: 2 }}>
          {Object.entries(status_counts).map(([status, count]) => (
            count > 0 && (
              <Chip 
                key={status}
                label={`${statusLabels[status] || status}: ${count}`}
                color={statusColors[status] || 'default'}
                variant="outlined"
                size="small"
              />
            )
          ))}
        </Box>

        {stats.document_sizes && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="textSecondary">
              Размеры файлов: 
              мин. {formatFileSize(stats.document_sizes.min)}, 
              макс. {formatFileSize(stats.document_sizes.max)}, 
              сред. {formatFileSize(stats.document_sizes.avg)}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Компонент для отображения информации о чанках
const ChunksCard = ({ stats }) => {
  if (!stats || !stats.chunks) return null;

  const { total_count, per_document } = stats.chunks;

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          <PieChartIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Фрагменты ({total_count})
        </Typography>
        <Divider sx={{ my: 1 }} />
        
        {per_document && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
              <b>Фрагментов на документ:</b>
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Минимум: {per_document.min || 'N/A'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Максимум: {per_document.max || 'N/A'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Среднее: {per_document.avg ? Math.round(per_document.avg) : 'N/A'}
            </Typography>
          </Box>
        )}

        {stats.chunking && stats.chunking.current_config && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
              <b>Настройки чанкинга:</b>
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Режим: {stats.chunking.current_config.mode || 'character'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Размер: {stats.chunking.current_config.chunk_size || 'N/A'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Перекрытие: {stats.chunking.current_config.chunk_overlap || 'N/A'}
            </Typography>
          </Box>
        )}

        {stats.chunking && stats.chunking.modes_usage && stats.chunking.modes_usage.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
              <b>Использование режимов:</b>
            </Typography>
            {stats.chunking.modes_usage.map((mode, index) => (
              <Typography key={index} variant="body2" color="textSecondary">
                {mode.mode}: {mode.count} документов
              </Typography>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Компонент для отображения информации о векторной базе
const VectorDbCard = ({ stats }) => {
  if (!stats || !stats.vector_db) return null;

  const { points_count, vectors_count, segments_count, storage_size, collection_name, error } = stats.vector_db;

  if (error) {
    return (
      <Card variant="outlined" sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Векторная база
          </Typography>
          <Divider sx={{ my: 1 }} />
          
          <Alert severity="error" sx={{ mt: 2 }}>
            Ошибка получения информации о векторной базе: {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
          Векторная база
        </Typography>
        <Divider sx={{ my: 1 }} />
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" color="textSecondary">
            Коллекция: {collection_name || 'default'}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Количество точек: {points_count || 0}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Количество векторов: {vectors_count || 0}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Количество сегментов: {segments_count || 0}
          </Typography>
          {storage_size !== undefined && (
            <Typography variant="body2" color="textSecondary">
              Размер хранилища: {formatFileSize(storage_size)}
            </Typography>
          )}
        </Box>

        {stats.search && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
              <b>Настройки поиска:</b>
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Модель эмбеддингов: {stats.search.model_name ? stats.search.model_name.split('/').pop() : 'N/A'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Размерность векторов: {stats.search.vector_size || 'N/A'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Гибридный поиск: {stats.search.use_hybrid ? 'Да' : 'Нет'}
            </Typography>
            {stats.search.use_hybrid && (
              <>
                <Typography variant="body2" color="textSecondary">
                  Вес векторного поиска: {stats.search.dense_weight || 0}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Вес лексического поиска: {stats.search.sparse_weight || 0}
                </Typography>
              </>
            )}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Компонент для отображения топ пользователей
const TopUsersCard = ({ stats }) => {
  if (!stats || !stats.users) return null;

  const { total, top_by_docs } = stats.users;

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Пользователи ({total})
        </Typography>
        <Divider sx={{ my: 1 }} />
        
        {top_by_docs && top_by_docs.length > 0 ? (
          <List dense disablePadding>
            {top_by_docs.map((user, index) => (
              <ListItem key={index} disablePadding sx={{ py: 0.5 }}>
                <ListItemText
                  primary={`${user.username}`}
                  secondary={`${user.docs_count} документов`}
                />
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography variant="body2" color="textSecondary">
            Нет данных о пользователях
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

// Форматирование размера файла
const formatFileSize = (size) => {
  if (size === null || size === undefined) return 'N/A';
  
  if (size < 1024) return size + ' B';
  if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB';
  if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed(1) + ' MB';
  return (size / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
};

// Основной компонент статистики
function SystemStats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/stats');
      setStats(response.data);
    } catch (err) {
      console.error('Error fetching stats:', err);
      setError(err?.response?.data?.detail || err.message || 'Ошибка получения статистики');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" component="h2" sx={{ flexGrow: 1 }}>
          Статистика системы
        </Typography>
        <Tooltip title="Обновить статистику">
          <IconButton onClick={fetchStats} disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      ) : (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <DocumentStatusCard stats={stats} />
          </Grid>
          <Grid item xs={12} md={6}>
            <ChunksCard stats={stats} />
          </Grid>
          <Grid item xs={12} md={6}>
            <VectorDbCard stats={stats} />
          </Grid>
          <Grid item xs={12} md={6}>
            <TopUsersCard stats={stats} />
          </Grid>
        </Grid>
      )}
    </Paper>
  );
}

export default SystemStats; 