import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Alert,
  Button,
  Snackbar,
  FormControlLabel,
  Switch
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import { getDocuments, uploadDocuments, reindexDocuments, deleteDocument, clearRagIndex, deleteAllDocuments } from '../services/api';
import FileUpload from '../components/FileUpload';
import DocumentList from '../components/DocumentList';
import { useAuth } from '../contexts/AuthContext';

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState('');
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });
  const [selectedIds, setSelectedIds] = useState([]);
  const [showAllDocs, setShowAllDocs] = useState(false);
  const { isAdmin } = useAuth();

  // Загрузка списка документов при монтировании или изменении фильтра
  useEffect(() => {
    fetchDocuments();
  }, [showAllDocs]);

  // Получение списка документов
  const fetchDocuments = async () => {
    setLoading(true);
    setError('');
   
    try {
      const docs = await getDocuments(isAdmin() && showAllDocs);
      setDocuments(docs);
    } catch (err) {
      setError('Не удалось загрузить список документов');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Загрузка нового документа
  const handleUploadDocument = async (titles, files) => {
    setUploading(true);
    setError('');
   
    try {
      await uploadDocuments(titles, files);
      // После успешной загрузки обновляем список документов
      await fetchDocuments();
      return true;
    } catch (err) {
      const errorMessage = err.message || 'Ошибка при загрузке документов';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  // Переиндексация документов
  const handleReindexDocuments = async () => {
    setReindexing(true);
    setError('');
   
    try {
      const result = await reindexDocuments(isAdmin() && showAllDocs);
      setNotification({
        open: true,
        message: `Переиндексировано ${result.indexed_documents} документов, создано ${result.total_chunks} фрагментов`,
        severity: 'success'
      });
    } catch (err) {
      const errorMessage = err.message || 'Ошибка при переиндексации документов';
      setError(errorMessage);
      setNotification({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setReindexing(false);
    }
  };

  // Очистка индекса векторной базы
  const handleClearIndex = async () => {
    if (!window.confirm('Вы действительно хотите очистить индекс векторной базы? Это удалит все данные из индекса, но записи документов в базе данных останутся.')) {
      return;
    }

    setClearing(true);
    setError('');
   
    try {
      const result = await clearRagIndex();
      setNotification({
        open: true,
        message: `Индекс векторной базы успешно очищен`,
        severity: 'success'
      });
    } catch (err) {
      const errorMessage = err.message || 'Ошибка при очистке индекса';
      setError(errorMessage);
      setNotification({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setClearing(false);
    }
  };

  // Удаление всех документов (только для админов)
  const handleDeleteAllDocuments = async () => {
    if (!window.confirm('ВНИМАНИЕ! Вы собираетесь удалить ВСЕ документы из системы. Это действие нельзя отменить. Продолжить?')) {
      return;
    }

    setClearing(true);
    setError('');
   
    try {
      const result = await deleteAllDocuments();
      setNotification({
        open: true,
        message: result.message || 'Все документы успешно удалены',
        severity: 'success'
      });
      await fetchDocuments();
    } catch (err) {
      const errorMessage = err.message || 'Ошибка при удалении всех документов';
      setError(errorMessage);
      setNotification({
        open: true,
        message: errorMessage,
        severity: 'error'
      });
    } finally {
      setClearing(false);
    }
  };

  // Удаление документа
  const handleDelete = async (docId) => {
    if (!window.confirm('Удалить документ?')) return;
    try {
      await deleteDocument(docId);
      await fetchDocuments();
      setNotification({ open: true, message: 'Документ удалён', severity: 'success' });
    } catch (err) {
      setError(err.message || 'Ошибка при удалении документа');
      setNotification({ open: true, message: err.message || 'Ошибка при удалении документа', severity: 'error' });
    }
  };

  // Закрытие уведомления
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  // Массовый выбор чекбоксов
  const handleSelect = (id, checked) => {
    setSelectedIds(prev => checked ? [...prev, id] : prev.filter(x => x !== id));
  };
  const handleSelectAll = (checked) => {
    setSelectedIds(checked ? documents.map(d => d.id) : []);
  };
  // Массовое удаление
  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!window.confirm(`Удалить выбранные документы (${selectedIds.length})?`)) return;
    try {
      await Promise.all(selectedIds.map(id => deleteDocument(id)));
      setNotification({ open: true, message: 'Выбранные документы удалены', severity: 'success' });
      setSelectedIds([]);
      await fetchDocuments();
    } catch (err) {
      setError(err.message || 'Ошибка при массовом удалении');
      setNotification({ open: true, message: err.message || 'Ошибка при массовом удалении', severity: 'error' });
    }
  };

  // Переключение отображения всех документов (только для админов)
  const toggleShowAllDocs = (event) => {
    setShowAllDocs(event.target.checked);
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Управление документами
        </Typography>
        
        <Box>
          {isAdmin() && (
            <Button
              variant="outlined"
              color="secondary"
              onClick={handleDeleteAllDocuments}
              disabled={clearing}
              sx={{ mr: 2 }}
            >
              Удалить все документы
            </Button>
          )}
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteSweepIcon />}
            onClick={handleClearIndex}
            disabled={clearing}
            sx={{ mr: 2 }}
          >
            Очистить индекс
          </Button>
          <Button
            variant="outlined"
            color="primary"
            startIcon={<RefreshIcon />}
            onClick={handleReindexDocuments}
            disabled={reindexing}
          >
            Переиндексировать
          </Button>
        </Box>
      </Box>
      
      {isAdmin() && (
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Switch 
                checked={showAllDocs} 
                onChange={toggleShowAllDocs}
                color="primary"
              />
            }
            label="Показать документы всех пользователей"
          />
        </Box>
      )}
      
      <Typography variant="body1" paragraph>
        Загружайте документы для использования в RAG-системе.
        Поддерживаются текстовые файлы, которые будут проиндексированы для поиска.
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      
      <Box sx={{ mb: 4 }}>
        <FileUpload
          onUpload={handleUploadDocument}
          loading={uploading}
        />
      </Box>
      
      <DocumentList
        documents={documents}
        loading={loading}
        onDelete={handleDelete}
        selectedIds={selectedIds}
        onSelect={handleSelect}
        onSelectAll={handleSelectAll}
        onDeleteSelected={handleDeleteSelected}
        showOwner={isAdmin() && showAllDocs}
      />
      
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        message={notification.message}
      />
    </Container>
  );
}

export default DocumentsPage;