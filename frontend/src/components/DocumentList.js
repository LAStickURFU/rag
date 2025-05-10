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
  Tooltip
} from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import WebIcon from '@mui/icons-material/Web';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import PersonIcon from '@mui/icons-material/Person';
import api from '../services/api';

function DocumentList({ 
  documents, 
  loading, 
  onDelete, 
  selectedIds = [], 
  onSelect, 
  onSelectAll, 
  onDeleteSelected,
  showOwner = false 
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
          Загруженные документы
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
      <List sx={{ width: '100%' }}>
        {documents.map((doc, index) => (
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
                    <Chip
                      label={doc.source}
                      size="small"
                      sx={{ mr: 1, fontSize: '0.75rem', mt: 0.5 }}
                      color="primary"
                      variant="outlined"
                    />
                    {showOwner && doc.uploader && (
                      <Tooltip title="Пользователь, загрузивший документ">
                        <Chip
                          icon={<PersonIcon />}
                          label={doc.uploader}
                          size="small"
                          sx={{ mr: 1, fontSize: '0.75rem', mt: 0.5 }}
                          color="secondary"
                          variant="outlined"
                        />
                      </Tooltip>
                    )}
                    <Box
                      component="span"
                      sx={{ mt: 0.5, display: 'inline-block', color: 'text.secondary', fontSize: '0.875rem' }}
                    >
                      Добавлен: {formatDate(doc.created_at)}
                    </Box>
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
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Статус:</b> {detail.status}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Имя файла:</b> {detail.filename || '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Размер файла:</b> {detail.size ? `${(detail.size/1024).toFixed(1)} KB` : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Дата загрузки:</b> {detail.created_at ? new Date(detail.created_at).toLocaleString() : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Последнее обновление:</b> {detail.updated_at ? new Date(detail.updated_at).toLocaleString() : '-'}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Источник:</b> {detail.source}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>UUID:</b> {detail.uuid}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>ID:</b> {detail.id}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Кол-во чанков:</b> {detail.chunks_count}</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Длина текста:</b> {detail.content_length} символов</Box>
              <Box component="span" sx={{ display: 'block', mb: 1 }}><b>Режим чанкинга:</b> {detail.chunking_mode || 'character'}</Box>
              {showOwner && detail.user_id && (
                <Box component="span" sx={{ display: 'block', mb: 1 }}><b>ID пользователя:</b> {detail.user_id}</Box>
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