import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Alert,
  Button,
  Snackbar,
  LinearProgress
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getDocuments, uploadDocuments, reindexDocuments, deleteDocument } from '../services/api';
import FileUpload from '../components/FileUpload';
import DocumentList from '../components/DocumentList';

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [error, setError] = useState('');
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });
  const [selectedIds, setSelectedIds] = useState([]);

  // Загрузка списка документов при монтировании
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Получение списка документов
  const fetchDocuments = async () => {
    setLoading(true);
    setError('');
   
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      setError('Не удалось загрузить список документов');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Загрузка нового документа
  const handleUploadDocument = async (titles, sources, files) => {
    setUploading(true);
    setError('');
   
    try {
      await uploadDocuments(titles, sources, files);
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
      const result = await reindexDocuments();
      setNotification({
        open: true,
        message: `${result.message}: ${result.indexed_documents || 0} документов, ${result.total_chunks || 0} фрагментов`,
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

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4" component="h1">
          Управление документами
        </Typography>
       
        <Button
          variant="outlined"
          color="primary"
          startIcon={<RefreshIcon />}
          onClick={handleReindexDocuments}
          disabled={reindexing || documents.length === 0}
        >
          Переиндексировать
        </Button>
      </Box>
     
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