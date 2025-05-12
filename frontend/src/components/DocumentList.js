import React, { useState } from 'react';
import {
  Typography,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  Chip,
  useTheme,
  CircularProgress,
  IconButton,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Divider,
  Tooltip,
  LinearProgress,
  Alert,
  Pagination
} from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import WebIcon from '@mui/icons-material/Web';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import PersonIcon from '@mui/icons-material/Person';
import api from '../services/api';

// Компонент для отображения статуса обработки документа
const DocumentStatus = ({ status }) => {
  // Определяем настройки отображения в зависимости от статуса
  const getStatusConfig = () => {
    switch (status) {
      case 'processing':
        return { 
          color: 'primary', 
          label: 'Извлечение текста...', 
          showProgress: true, 
          progress: 20 
        };
      case 'chunking':
        return { 
          color: 'primary', 
          label: 'Разбиение на фрагменты...', 
          showProgress: true, 
          progress: 40 
        };
      case 'embedding':
        return { 
          color: 'primary', 
          label: 'Создание эмбеддингов...', 
          showProgress: true, 
          progress: 70 
        };
      case 'indexing':
        return { 
          color: 'primary', 
          label: 'Индексация документа...', 
          showProgress: true, 
          progress: 90 
        };
      case 'reindexing':
        return { 
          color: 'secondary', 
          label: 'Переиндексация...', 
          showProgress: true, 
          progress: 50 
        };
      case 'indexed':
        return { 
          color: 'success', 
          label: 'Обработан', 
          showProgress: false 
        };
      case 'error':
        return { 
          color: 'error', 
          label: 'Ошибка обработки', 
          showProgress: false 
        };
      case 'uploaded':
        return { 
          color: 'info', 
          label: 'Загружен, ожидает обработки', 
          showProgress: true,
          progress: 10
        };
      default:
        return { 
          color: 'info', 
          label: status || 'Неизвестный статус', 
          showProgress: false 
        };
    }
  };

  const config = getStatusConfig();

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', maxWidth: 200 }}>
      <Chip
        label={config.label}
        size="small"
        color={config.color}
        variant="outlined"
        sx={{ mt: 0.5, mb: config.showProgress ? 1 : 0 }}
      />
      {config.showProgress && (
        <LinearProgress 
          variant="determinate" 
          value={config.progress} 
          sx={{ height: 5, borderRadius: 5 }} 
        />
      )}
    </Box>
  );
};

function DocumentList({ 
  documents, 
  loading, 
  onDelete, 
  selectedIds = [], 
  onSelect, 
  onSelectAll, 
  onDeleteSelected,
  showOwner = false,
  pagination = null,
  onPageChange = null
}) {
  const theme = useTheme();
  const [detailOpen, setDetailOpen] = useState(false);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');

  // Выбор иконки в зависимости от источника документа
  const getDocumentIcon = (source) => {
    switch (source) {
      case 'web':
        return <WebIcon />;
      case 'pdf':
        return <PictureAsPdfIcon />;
      case 'manual':
        return <DescriptionIcon />;
      default:
        return <InsertDriveFileIcon />;
    }
  };

  // Форматирование даты
  const formatDate = (dateString) => {
    const options = {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('ru-RU', options);
  };

  const handleOpenDetail = async (docId) => {
    setDetailOpen(true);
    setDetail(null);
    setDetailLoading(true);
    setDetailError('');
    try {
      const res = await api.get(`/documents/${docId}`);
      setDetail(res.data);
    } catch (e) {
      setDetailError(e?.response?.data?.detail || e.message);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleCloseDetail = () => {
    setDetailOpen(false);
    setDetail(null);
    setDetailError('');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Загруженные документы
        </Typography>
        <Typography variant="body1" color="textSecondary" align="center" sx={{ py: 3 }}>
          У вас пока нет загруженных документов.
        </Typography>
      </Paper>
    );
  }

  const allSelected = documents.length > 0 && selectedIds.length === documents.length;
  const selectedCount = selectedIds.length;

  // Вычисляем, есть ли документы в процессе обработки
  const hasProcessingDocuments = Array.isArray(documents) && documents.some(doc => 
    ['processing', 'chunking', 'embedding', 'indexing', 'uploaded', 'reindexing'].includes(doc.status)
  );

  // Проверяем, идет ли переиндексация
  const isReindexing = Array.isArray(documents) && documents.some(doc => doc.status === 'reindexing');

  // Вычисляем общий прогресс обработки всех документов
  const calculateTotalProgress = () => {
    if (!hasProcessingDocuments || !Array.isArray(documents)) return 100;

    const totalDocs = documents.length;
    let progressSum = 0;
    
    documents.forEach(doc => {
      if (doc.status === 'indexed' || doc.status === 'error') {
        progressSum += 100;
      } else {
        // Определяем процент готовности для документа в обработке
        switch (doc.status) {
          case 'uploaded':
            progressSum += 5;
            break;
          case 'processing':
            progressSum += 25;
            break;
          case 'chunking':
            progressSum += 50;
            break;
          case 'embedding':
            progressSum += 75;
            break;
          case 'indexing':
            progressSum += 90;
            break;
          case 'reindexing':
            progressSum += 30; // Переиндексация - примерно 30% готовности
            break;
          default:
            progressSum += 0;
        }
      }
    });
    
    return totalDocs > 0 ? Math.floor(progressSum / totalDocs) : 0;
  };

  const totalProgress = calculateTotalProgress();

  return (
    <Paper elevation={2} sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', p: 2, pb: 1 }}>
        <Checkbox
          checked={allSelected}
          indeterminate={selectedCount > 0 && !allSelected}
          onChange={e => onSelectAll && onSelectAll(e.target.checked)}
          sx={{ mr: 1 }}
        />
        <Typography variant="h6" sx={{ flex: 1 }}>
          Загруженные документы {showOwner && ' (все пользователи)'}
        </Typography>
        {onDeleteSelected && (
          <Box>
            <IconButton
              color="error"
              onClick={onDeleteSelected}
              disabled={selectedCount === 0}
              sx={{ ml: 1 }}
              title={selectedCount > 0 ? `Удалить выбранные (${selectedCount})` : 'Нет выбранных'}
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        )}
      </Box>
      
      {/* Индикатор общего прогресса обработки */}
      {hasProcessingDocuments ? (
        <Box sx={{ px: 2, pb: 2 }}>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
            Общий прогресс: {totalProgress}%
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={totalProgress} 
            sx={{ height: 10, borderRadius: 5 }} 
          />
        </Box>
      ) : (
        Array.isArray(documents) && documents.some(doc => doc.status === 'error') ? (
          <Box sx={{ px: 2, py: 1, mb: 1 }}>
            <Alert severity="warning">
              Некоторые документы не удалось обработать. Проверьте детали, нажав на документ.
            </Alert>
          </Box>
        ) : documents.length > 0 ? (
          <Box sx={{ px: 2, py: 1, mb: 1 }}>
            <Alert severity="success">
              Все документы успешно обработаны и проиндексированы.
            </Alert>
          </Box>
        ) : null
      )}
      
      {/* Предупреждение о документах в обработке */}
      {hasProcessingDocuments && (
        <Box sx={{ px: 2, py: 1, mb: 1 }}>
          <Alert severity="info">
            {isReindexing ? 
              "Идет переиндексация документов. Страница автоматически обновляется, ожидайте завершения." : 
              "Некоторые документы находятся в процессе обработки. Страница автоматически обновляется, ожидайте завершения обработки."
            }
          </Alert>
        </Box>
      )}
      
      <List sx={{ width: '100%' }}>
        {Array.isArray(documents) && documents.map((doc, index) => (
          <React.Fragment key={doc.id}>
            {index > 0 && <li><hr /></li>}
            <ListItem alignItems="flex-start" sx={{ py: 2 }}
              button
              onClick={() => handleOpenDetail(doc.id)}
              selected={detail && detail.id === doc.id}
            >
              <Checkbox
                checked={selectedIds.includes(doc.id)}
                onChange={e => onSelect && onSelect(doc.id, e.target.checked)}
                sx={{ mr: 1 }}
              />
              <ListItemIcon sx={{ color: theme.palette.primary.main }}>
                {getDocumentIcon(doc.source)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Typography
                    variant="subtitle1"
                    sx={{ fontWeight: 500 }}
                    component="div"
                  >
                    {doc.title}
                  </Typography>
                }
                secondary={
                  <React.Fragment>
                    <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                      <Chip
                        label={doc.source || 'manual_upload'}
                        size="small"
                        sx={{ fontSize: '0.75rem' }}
                        color="primary"
                        variant="outlined"
                      />
                      {showOwner && doc.uploader && (
                        <Tooltip title="Владелец документа">
                          <Chip
                            icon={<PersonIcon />}
                            label={doc.uploader}
                            size="small"
                            sx={{ fontSize: '0.75rem' }}
                            color="secondary"
                            variant="outlined"
                          />
                        </Tooltip>
                      )}
                      
                      {/* Добавляем индикатор статуса */}
                      <DocumentStatus status={doc.status} />
                    </Box>
                    
                    <Box
                      component="span"
                      sx={{ mt: 1, display: 'block', color: 'text.secondary', fontSize: '0.875rem' }}
                    >
                      Добавлен: {formatDate(doc.created_at)}
                    </Box>
                    
                    {doc.chunks_count > 0 && (
                      <Box
                        component="span"
                        sx={{ display: 'block', color: 'text.secondary', fontSize: '0.875rem' }}
                      >
                        Фрагментов: {doc.chunks_count}
                      </Box>
                    )}
                    
                    {/* Добавляем основную информацию о чанкинге и для списка */}
                    {doc.chunking_mode && (
                      <Box
                        component="span"
                        sx={{ display: 'block', color: 'text.secondary', fontSize: '0.875rem' }}
                      >
                        Режим чанкинга: {doc.chunking_mode}
                        {doc.processing_summary?.chunk_size && ` | Размер: ${doc.processing_summary.chunk_size}`}
                        {doc.processing_summary?.chunk_overlap && ` | Перекрытие: ${doc.processing_summary.chunk_overlap}`}
                      </Box>
                    )}
                  </React.Fragment>
                }
                secondaryTypographyProps={{ component: 'div' }}
              />
              {onDelete && (
                <IconButton edge="end" aria-label="delete" onClick={(e) => {
                  e.stopPropagation();
                  onDelete(doc.id);
                }}>
                  <DeleteIcon />
                </IconButton>
              )}
            </ListItem>
          </React.Fragment>
        ))}
      </List>
      
      {/* Добавляем компонент пагинации, если она доступна */}
      {pagination && pagination.total_pages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <Pagination 
            count={pagination.total_pages} 
            page={pagination.page + 1} // API использует 0-based индексы, Pagination использует 1-based
            onChange={(e, page) => onPageChange && onPageChange(page - 1)} // Конвертируем обратно в 0-based
            color="primary"
            showFirstButton
            showLastButton
          />
        </Box>
      )}
      
      {/* Диалог подробной информации */}
      <Dialog open={detailOpen} onClose={handleCloseDetail} maxWidth="sm" fullWidth>
        <DialogTitle>Информация о документе</DialogTitle>
        <DialogContent dividers>
          {detailLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : detailError ? (
            <Typography color="error">{detailError}</Typography>
          ) : detail ? (
            <Box>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>{detail.title}</Typography>
              <Divider sx={{ mb: 1 }} />
              
              <Box sx={{ mb: 2 }}>
                <DocumentStatus status={detail.status} />
              </Box>
              
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Имя файла:</b> {detail.filename || '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Размер файла:</b> {detail.size ? `${(detail.size/1024).toFixed(1)} KB` : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Дата загрузки:</b> {detail.created_at ? new Date(detail.created_at).toLocaleString() : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Последнее обновление:</b> {detail.updated_at ? new Date(detail.updated_at).toLocaleString() : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Источник:</b> {detail.source}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>UUID:</b> {detail.uuid}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>ID:</b> {detail.id}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Кол-во чанков:</b> {detail.chunks_count}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Длина текста:</b> {detail.content_length} символов</Box>
              
              {/* Расширенная информация о параметрах индексации */}
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Параметры индексации:</Typography>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Режим чанкинга:</b> {detail.chunking_mode || 'character'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Размер чанка:</b> {detail.chunking_info?.chunk_size || '-'} токенов</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Перекрытие чанков:</b> {detail.chunking_info?.chunk_overlap || '-'} токенов</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Модель эмбеддингов:</b> {detail.chunking_info?.embedding_model || '-'}</Box>
              
              {/* Дополнительные параметры обработки */}
              {detail.processing_info && Object.keys(detail.processing_info).length > 0 && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Typography variant="subtitle2" sx={{ mb: 1 }}>Дополнительные параметры:</Typography>
                  {detail.processing_info.use_hybrid_search !== undefined && (
                    <Box component="span" sx={{ display: 'block', mb: 1 }}>
                      <b>Гибридный поиск:</b> {detail.processing_info.use_hybrid_search ? 'Да' : 'Нет'}
                    </Box>
                  )}
                  {detail.processing_info.dense_weight !== undefined && (
                    <Box component="span" sx={{ display: 'block', mb: 1 }}>
                      <b>Вес векторного поиска:</b> {detail.processing_info.dense_weight}
                    </Box>
                  )}
                  {detail.processing_info.sparse_weight !== undefined && (
                    <Box component="span" sx={{ display: 'block', mb: 1 }}>
                      <b>Вес лексического поиска:</b> {detail.processing_info.sparse_weight}
                    </Box>
                  )}
                  {detail.processing_info.reranker_weight !== undefined && (
                    <Box component="span" sx={{ display: 'block', mb: 1 }}>
                      <b>Вес ранжирования:</b> {detail.processing_info.reranker_weight}
                    </Box>
                  )}
                </>
              )}
              
              {showOwner && detail.uploader && (
                <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Владелец:</b> {detail.uploader} (ID: {detail.user_id})</Box>
              )}
              {detail.error_message && (
                <Box component="span" sx={{ display: 'block', color: 'error.main', mb: 1 }}><b>Ошибка:</b> {detail.error_message}</Box>
              )}
            </Box>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDetail}>Закрыть</Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
}

export default DocumentList;