import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Alert,
  Button,
  Snackbar,
  CircularProgress,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  MenuItem,
  Select,
  FormControl,
  InputLabel
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getDocuments, uploadDocuments, reindexDocuments, deleteDocument, clearRagIndex, deleteAllDocuments } from '../services/api';
import FileUpload from '../components/FileUpload';
import DocumentList from '../components/DocumentList';
import DocumentStats from '../components/DocumentStats';
import { useAuth } from '../contexts/AuthContext';
import ReindexConfirmDialog from '../components/ReindexConfirmDialog';
import api from '../services/api';

function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [error, setError] = useState('');
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });
  const [selectedIds, setSelectedIds] = useState([]);
  const [showAllDocs, setShowAllDocs] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(false);
  const { isAdmin } = useAuth();
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [confirmReindexOpen, setConfirmReindexOpen] = useState(false);
  const [showAllUsers, setShowAllUsers] = useState(false);
  
  // Для отслеживания первого рендера при изменении фильтра
  const isFirstRenderRef = useRef(true);
  
  // Добавляем состояние для хранения данных пагинации
  const [pagination, setPagination] = useState({
    page: 0,
    page_size: 20,
    total: 0,
    total_pages: 0,
    has_next: false,
    has_prev: false
  });
  
  // Проверяем, есть ли документы в процессе обработки
  const hasProcessingDocuments = Array.isArray(documents) && documents.some(doc => 
    ['processing', 'chunking', 'embedding', 'indexing', 'uploaded', 'reindexing'].includes(doc.status)
  );

  // Преобразуем fetchDocuments в useCallback для использования в эффектах
  const fetchDocuments = useCallback(async (page = pagination.page) => {
    console.log(`Запрос документов: page=${page}, showAllUsers=${showAllUsers}, режим=${refreshInterval ? 'автообновление' : 'обычный'}`);
    setError('');
    try {
      // В режиме автообновления всегда запрашиваем все документы без пагинации
      // Если документы в процессе обработки, также запрашиваем все
      const useReturnAll = refreshInterval;
      
      const response = await getDocuments(
        isAdmin() && showAllUsers,
        useReturnAll ? 0 : page, 
        pagination.page_size,
        useReturnAll // returnAll=true для автообновления
      );
      
      if (useReturnAll) {
        // Старый формат ответа - просто массив документов
        setDocuments(response);
        return response;
      } else {
        // Новый формат ответа с пагинацией
        setDocuments(response.items);
        setPagination(response.pagination);
        return response.items;
      }
    } catch (err) {
      setError('Не удалось загрузить список документов');
      console.error(err);
      return [];
    } finally {
      setLoading(false);
    }
  }, [isAdmin, showAllUsers, refreshInterval, pagination.page_size, pagination.page]);

  // Отдельный эффект для только одной цели - начальной загрузки страницы
  useEffect(() => {
    console.log('Первоначальная загрузка документов');
    setLoading(true);
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Преднамеренно пустой массив зависимостей

  // Обработчик изменения страницы пагинации
  const handlePageChange = (newPage) => {
    console.log(`Изменена страница пагинации на ${newPage}`);
    // Обновляем страницу и загружаем документы для этой страницы
    setPagination(prev => ({ ...prev, page: newPage }));
    setLoading(true);
    fetchDocuments(newPage);
  };

  // Эффект для отслеживания изменения фильтра showAllUsers
  useEffect(() => {
    // Не запускаем при первом рендере
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false;
      return;
    }
    
    console.log('Изменился фильтр showAllUsers, запрашиваем документы');
    setLoading(true);
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showAllUsers]); // Только showAllUsers как зависимость

  // Эффект для интервального обновления
  useEffect(() => {
    // Пропускаем, если нет необходимости в автообновлении
    if (!hasProcessingDocuments && !refreshInterval) {
      return;
    }
    
    console.log("Настройка интервального обновления документов");
    
    // Создаем интервал для автоматического обновления каждые 2 секунды
    const autoRefreshInterval = setInterval(() => {
      // Проверяем, есть ли документы в обработке
      const hasProcessingDocs = Array.isArray(documents) && documents.some(doc => 
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasProcessingDocuments, refreshInterval, documents]); // Только необходимые зависимости

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

  // Переиндексация документов
  const handleReindexDocuments = async (includeAllUsers = false) => {
    setReindexing(true);
    setError('');
   
    try {
      // Переиндексируем только документы всех пользователей, если пользователь админ и выбран соответствующий параметр
      const reindexAllUsers = isAdmin() && includeAllUsers;
      const result = await reindexDocuments(reindexAllUsers);
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

    setLoading(true);
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
      setLoading(false);
    }
  };

  // Удаление всех документов (только для админов)
  const handleDeleteAllDocuments = async () => {
    if (!window.confirm('ВНИМАНИЕ! Вы собираетесь удалить ВСЕ документы из системы. Это действие нельзя отменить. Продолжить?')) {
      return;
    }

    setLoading(true);
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
      setLoading(false);
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

  // Массовый выбор чекбоксов
  const handleSelect = (id, checked) => {
    setSelectedIds(prev => checked ? [...prev, id] : prev.filter(x => x !== id));
  };
  const handleSelectAll = (checked) => {
    setSelectedIds(checked && Array.isArray(documents) ? documents.map(d => d.id) : []);
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

  // Обновляем обработчик ручного обновления списка документов
  const handleRefresh = async () => {
    console.log('Ручное обновление списка документов');
    // Предотвращаем повторное нажатие во время загрузки
    if (!loading) {
      setLoading(true);
      await fetchDocuments();
    }
  };

  // Обновляем обработчик изменения выбора в выпадающем списке
  const handleShowAllUsersChange = (e) => {
    console.log('Выбран новый режим показа документов:', e.target.value);
    setShowAllUsers(e.target.value);
    // Обновление документов будет выполнено эффектом выше
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Управление документами
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2 }}>
          {isAdmin() && (
            <FormControl variant="outlined" size="small" sx={{ minWidth: 200 }}>
              <InputLabel id="show-all-users-label">Показать документы</InputLabel>
              <Select
                labelId="show-all-users-label"
                value={showAllUsers}
                onChange={handleShowAllUsersChange}
                label="Показать документы"
              >
                <MenuItem value={false}>Только мои</MenuItem>
                <MenuItem value={true}>Всех пользователей</MenuItem>
              </Select>
            </FormControl>
          )}
          
          <Box sx={{ display: 'flex', gap: 1, ml: isAdmin() ? 2 : 0 }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleRefresh}
              disabled={loading}
              size="medium"
            >
              ОБНОВИТЬ
            </Button>
            
            <Button
              variant="contained"
              color="secondary"
              onClick={() => setConfirmReindexOpen(true)}
              disabled={loading}
              size="medium"
            >
              ПЕРЕИНДЕКСИРОВАТЬ
            </Button>
            
            {isAdmin() && (
              <Button
                variant="contained"
                color="warning"
                onClick={handleClearIndex}
                disabled={loading}
                size="medium"
              >
                ОЧИСТИТЬ ИНДЕКС
              </Button>
            )}
            
            {isAdmin() && (
              <Button
                variant="contained"
                color="error"
                onClick={handleDeleteAllDocuments}
                disabled={loading}
                size="medium"
              >
                УДАЛИТЬ ВСЕ ДОКУМЕНТЫ
              </Button>
            )}
          </Box>
        </Box>
      </Box>
      
      {/* Добавляем компонент статистики */}
      <DocumentStats documentStats={documents} loading={loading} />
      
      <Typography variant="body1" paragraph>
        Загружайте документы для использования в RAG-системе.
        Поддерживаются текстовые файлы, которые будут проиндексированы для поиска.
      </Typography>
      
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
            onClick={handleRefresh}
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
        showOwner={isAdmin() && showAllUsers}
        pagination={!refreshInterval && !hasProcessingDocuments && Array.isArray(documents) && documents.length > 0 ? pagination : null}
        onPageChange={handlePageChange}
      />
      
      {/* Диалоги подтверждения */}
      <Dialog
        open={confirmDeleteOpen}
        onClose={() => setConfirmDeleteOpen(false)}
      >
        <DialogTitle>Подтверждение удаления</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Вы действительно хотите удалить {selectedIds.length > 1 
              ? `выбранные ${selectedIds.length} документов` 
              : 'выбранный документ'}?
            Эта операция не может быть отменена.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDeleteOpen(false)}>Отмена</Button>
          <Button onClick={() => {
            handleDeleteSelected();
            setConfirmDeleteOpen(false);
          }} color="error">Удалить</Button>
        </DialogActions>
      </Dialog>
      
      <ReindexConfirmDialog
        open={confirmReindexOpen}
        onClose={() => setConfirmReindexOpen(false)}
        onConfirm={handleReindexDocuments}
        showAllUsersOption={isAdmin()}
        allUsersSelected={showAllUsers}
      />
      
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
      >
        <Alert onClose={() => setNotification({ ...notification, open: false })} severity={notification.severity || 'info'}>
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default DocumentsPage;