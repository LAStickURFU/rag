import React, { useState, useRef } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  CircularProgress,
  Alert
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function FileUpload({ onUpload, loading, disabled }) {
  const [titles, setTitles] = useState([]);
  const [files, setFiles] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const formRef = useRef(null);
  const isSubmittingRef = useRef(false); // Доп. флаг для предотвращения повторной отправки

  const SUPPORTED_EXTENSIONS = [
    'doc', 'docx', 'txt', 'pdf', 'rtf', 'md', 'json', 'csv', 'tsv', 'html', 'htm', 'xml', 'yaml', 'yml', 'odt', 'epub', 'log', 'tex', 'ini', 'jsonl'
  ];
  const ACCEPT_STRING = SUPPORTED_EXTENSIONS.map(ext => '.' + ext).join(',');

  const handleFileChange = (e) => {
    if (disabled) return;
    
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;
    
    // Сразу очищаем предыдущий выбор файлов и названий
    let validFiles = [];
    let newTitles = [];
    
    for (const file of selectedFiles) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (!SUPPORTED_EXTENSIONS.includes(ext)) {
        setError(`Формат файла ${file.name} не поддерживается. Разрешённые форматы: ${SUPPORTED_EXTENSIONS.join(', ')}`);
        continue;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError(`Размер файла ${file.name} превышает 10MB`);
        continue;
      }
      validFiles.push(file);
      newTitles.push(file.name.split('.')[0]);
    }
    
    if (validFiles.length === 0) return;
    
    // Заменяем старые файлы и названия новыми
    setFiles(validFiles);
    setTitles(newTitles);
    setError('');
    
    // Сбрасываем статус успешной загрузки
    setSuccess(false);
  };

  const handleDrag = (e) => {
    if (disabled) return;
    
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    if (disabled) return;
    
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleTitleChange = (idx, value) => {
    if (disabled) return;
    
    const arr = [...titles];
    arr[idx] = value;
    setTitles(arr);
  };
  
  const handleRemoveFile = (idx) => {
    if (disabled) return;
    
    setFiles(files.filter((_, i) => i !== idx));
    setTitles(titles.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    e.stopPropagation(); // Добавляем остановку всплытия событий
    
    if (disabled) return;
    
    // Двойная проверка на повторную отправку
    if (submitting || loading || isSubmittingRef.current) {
      console.log('Форма уже отправляется или идет загрузка');
      return;
    }
    
    // Устанавливаем флаг отправки сразу в обоих местах
    setSubmitting(true);
    isSubmittingRef.current = true;
    
    setError('');
    setSuccess(false);
    
    if (!files.length) {
      setError('Выберите хотя бы один файл для загрузки');
      setSubmitting(false);
      isSubmittingRef.current = false;
      return;
    }
    
    for (let i = 0; i < titles.length; ++i) {
      if (!titles[i] || !titles[i].trim()) {
        setError('Укажите название для всех документов');
        setSubmitting(false);
        isSubmittingRef.current = false;
        return;
      }
    }
    
    try {
      console.log('Отправка файлов на сервер:', files.length, 'файлов');
      // Блокируем кнопку submit на форме для предотвращения повторных нажатий
      if (formRef.current) {
        const submitButton = formRef.current.querySelector('[type="submit"]');
        if (submitButton) submitButton.disabled = true;
      }
      
      // Используем только функцию onUpload, предоставленную родительским компонентом
      await onUpload(titles, files);
      setSuccess(true);
      setTitles([]);
      setFiles([]);
      
      // Сбросить input
      const fileInput = document.getElementById('document-file');
      if (fileInput) fileInput.value = '';
    } catch (err) {
      console.error('Ошибка при загрузке:', err);
      setError(err.message || 'Ошибка при загрузке файлов');
    } finally {
      setSubmitting(false);
      isSubmittingRef.current = false;
    }
  };

  // Проверяем, отключена ли форма
  const isFormDisabled = loading || submitting || disabled;

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Загрузить новые документы
      </Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }} data-testid="upload-error">{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>Документы успешно загружены!</Alert>}
      {disabled && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Загрузка новых документов временно недоступна, так как система обрабатывает существующие документы. 
          Пожалуйста, дождитесь завершения текущих процессов.
        </Alert>
      )}
      <Box
        component="form"
        onSubmit={handleSubmit}
        onDragEnter={handleDrag}
        sx={{ position: 'relative', opacity: isFormDisabled ? 0.7 : 1 }}
        ref={formRef}
      >
        <Box
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          sx={{
            border: dragActive ? '2px dashed #1976d2' : '2px dashed #ccc',
            borderRadius: 2,
            p: 3,
            mb: 2,
            textAlign: 'center',
            bgcolor: dragActive ? '#e3f2fd' : 'inherit',
            transition: 'background 0.2s',
            cursor: isFormDisabled ? 'not-allowed' : 'pointer',
            opacity: isFormDisabled ? 0.7 : 1,
          }}
        >
          <CloudUploadIcon color={dragActive ? 'primary' : 'disabled'} sx={{ fontSize: 48, mb: 1 }} />
          <Typography variant="body1" color={dragActive ? 'primary' : 'textSecondary'}>
            Перетащите файлы сюда или выберите вручную (можно несколько)
          </Typography>
          <Button
            variant="outlined"
            component="label"
            fullWidth
            startIcon={<UploadFileIcon />}
            sx={{ mt: 2, mb: 1 }}
            disabled={isFormDisabled}
          >
            Выбрать файлы
            <input
              id="document-file"
              type="file"
              accept={ACCEPT_STRING}
              hidden
              multiple
              onChange={handleFileChange}
              disabled={isFormDisabled}
              data-testid="file-input"
            />
          </Button>
        </Box>
        {files.length > 0 && files.map((file, idx) => (
          <Box key={idx} sx={{ mb: 2, p: 1, border: '1px solid #eee', borderRadius: 1, display: 'flex', alignItems: 'center' }}>
            <Typography variant="body2" sx={{ flex: 1 }}>
              {file.name} ({(file.size / 1024).toFixed(2)} KB)
            </Typography>
            <TextField
              label="Название"
              value={titles[idx] || ''}
              onChange={e => handleTitleChange(idx, e.target.value)}
              size="small"
              sx={{ mx: 1, width: 180 }}
              required
              inputProps={{ 'data-testid': 'title-input' }}
              disabled={isFormDisabled}
            />
            <Box data-testid="source-select" sx={{ display: 'none' }} />
            <Button 
              color="error" 
              size="small" 
              onClick={() => handleRemoveFile(idx)} 
              sx={{ ml: 1 }}
              disabled={isFormDisabled}
            >
              Удалить
            </Button>
          </Box>
        ))}
        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          disabled={isFormDisabled || !files.length}
          sx={{ mt: 2 }}
          startIcon={isFormDisabled && <CircularProgress size={20} color="inherit" />}
          data-testid="submit-button"
        >
          {isFormDisabled ? 'Загрузка...' : 'Загрузить документы'}
        </Button>
      </Box>
    </Paper>
  );
}

export default FileUpload;