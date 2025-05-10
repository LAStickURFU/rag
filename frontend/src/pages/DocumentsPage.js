import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Typography,
  Box,
  Alert,
  Button,
  Snackbar,
  FormControlLabel,
  Switch,
  CircularProgress,
  Paper
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
  const [refreshInterval, setRefreshInterval] = useState(false);
  const { isAdmin } = useAuth();

  // Преобразуем fetchDocuments в useCallback для использования в эффектах
  const fetchDocuments = useCallback(async () => {
    setError('');
    try {
      const docs = await getDocuments(isAdmin() && showAllDocs);
      setDocuments(docs);
      return docs;
    } catch (err) {
      setError('Не удалось загрузить список документов');
      console.error(err);
      return [];
    } finally {
      setLoading(false);
    }
  }, [isAdmin, showAllDocs]);

  // Отдельный эффект для автоматического обновления при загрузке страницы
  useEffect(() => {
    // При первой загрузке получаем документы
    setLoading(true);
    fetchDocuments();
  }, [fetchDocuments]);

  // Отдельный эффект для интервального обновления
  useEffect(() => {
    console.log("Настройка интервального обновления документов");
    
    // Создаем интервал для автоматического обновления каждые 2 секунды
    const autoRefreshInterval = setInterval(() => {
      // Проверяем, есть ли документы в обработке
      const hasProcessingDocs = documents.some(doc => 
        ['processing', 'chunking', 'embedding', 'indexing', 'uploaded', 'reindexing'].includes(doc.status)
      );
      
      // Логируем для отладки
      console.log(`Проверка документов: ${hasProcessingDocs ? 'есть в обработке' : 'все готовы'}, refreshInterval: ${refreshInterval}`);
      
      if (hasProcessingDocs) {
        // Если документы обрабатываются, обновляем список
        console.log('Автообновление: есть документы в обработке');
        fetchDocuments();
        
        // Показываем уведомление при первом обнаружении документов в обработке
        if (!refreshInterval) {
          console.log('Показываем уведомление о начале обработки');
          setNotification({
            open: true,
            message: 'Документы обрабатываются. Страница автоматически обновляется.',
            severity: 'info'
          });
          setRefreshInterval(true);
        }
      } else if (refreshInterval) {
        // Если все документы обработаны, но интервал был активен
        console.log('Автообновление: все документы обработаны, останавливаем');
        setNotification({
          open: true,
          message: 'Все документы обработаны и проиндексированы',
          severity: 'success'
        });
        setRefreshInterval(false);
      }
    }, 2000); // Интервал 2 секунды
    
    // Очистка интервала при размонтировании
    return () => {
      console.log("Очистка интервала обновления документов");
      clearInterval(autoRefreshInterval);
    };
  }, [documents, refreshInterval, fetchDocuments]); // documents изменяется при каждом fetchDocuments

  // Загрузка нового документа
  const handleUploadDocument = async (titles, files) => {
    setUploading(true);
    setError('');
   
    try {
      await uploadDocuments(titles, files);
      // После успешной загрузки обновляем список документов
      await fetchDocuments();
      
      // Показываем уведомление о начале обработки
      setNotification({
        open: true,
        message: 'Документы загружены и отправлены на обработку',
        severity: 'info'
      });
      
      return true;
    } catch (err) {
      const errorMessage = err.message || 'Ошибка при загрузке документов';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  // Проверяем, есть ли документы в процессе обработки
  const hasProcessingDocuments = documents.some(doc => 
    ['processing', 'chunking', 'embedding', 'indexing', 'uploaded', 'reindexing'].includes(doc.status)
  );

  // Переиндексация документов
  const handleReindexDocuments = async () => {
    setReindexing(true);
    setError('');
   
    try {
      const result = await reindexDocuments(isAdmin() && showAllDocs);
      console.log('Запущена переиндексация документов:', result);
      
      setNotification({
        open: true,
        message: result.message || 'Запущена переиндексация документов',
        severity: 'info'
      });
      
      // Сразу обновляем список документов, чтобы увидеть статус "reindexing"
      setLoading(true);
      await fetchDocuments();
      
      // Запускаем автоматическое обновление, если оно не активно
      setRefreshInterval(true);
      
      console.log('Запущено автоматическое обновление для отслеживания переиндексации');
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
      await clearRagIndex();
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

  // Ручное обновление списка документов
  const handleRefreshDocuments = async () => {
    setLoading(true);
    await fetchDocuments();
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
              disabled={clearing || hasProcessingDocuments}
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
            disabled={clearing || hasProcessingDocuments}
            sx={{ mr: 2 }}
          >
            Очистить индекс
          </Button>
          <Button
            variant="outlined"
            color="primary"
            startIcon={<RefreshIcon />}
            onClick={handleReindexDocuments}
            disabled={reindexing || hasProcessingDocuments}
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
      
      {/* Отображение прогресса автоматического обновления */}
      {hasProcessingDocuments && refreshInterval && (
        <Paper elevation={1} sx={{ p: 2, mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CircularProgress size={24} sx={{ mr: 2 }} />
            <Typography>Обработка документов в процессе... Страница обновляется автоматически</Typography>
          </Box>
          <Button 
            variant="text" 
            color="primary"
            onClick={handleRefreshDocuments}
            startIcon={<RefreshIcon />}
          >
            Обновить
          </Button>
        </Paper>
      )}
      
      <Box sx={{ mb: 4 }}>
        <FileUpload
          onUpload={handleUploadDocument}
          loading={uploading}
          disabled={hasProcessingDocuments}
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
        showOwner={isAdmin()}
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