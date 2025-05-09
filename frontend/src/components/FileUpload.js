import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Typography,
  Paper,
  CircularProgress,
  Alert
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

function FileUpload({ onUpload, loading }) {
  const [titles, setTitles] = useState([]);
  const [files, setFiles] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const SUPPORTED_EXTENSIONS = [
    'doc', 'docx', 'txt', 'pdf', 'rtf', 'md', 'json', 'csv', 'tsv', 'html', 'htm', 'xml', 'yaml', 'yml', 'odt', 'epub', 'log', 'tex', 'ini', 'jsonl'
  ];
  const ACCEPT_STRING = SUPPORTED_EXTENSIONS.map(ext => '.' + ext).join(',');

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    let validFiles = [];
    let newTitles = [...titles];
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
    if (validFiles.length === 0 && selectedFiles.length > 0) return;
    setFiles(prev => [...prev, ...validFiles]);
    setTitles(newTitles);
    setError('');
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleTitleChange = (idx, value) => {
    const arr = [...titles];
    arr[idx] = value;
    setTitles(arr);
  };
  const handleRemoveFile = (idx) => {
    setFiles(files.filter((_, i) => i !== idx));
    setTitles(titles.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess(false);
    if (!files.length) {
      setError('Выберите хотя бы один файл для загрузки');
      return;
    }
    for (let i = 0; i < titles.length; ++i) {
      if (!titles[i] || !titles[i].trim()) {
        setError('Укажите название для всех документов');
        return;
      }
    }
    try {
      await onUpload(titles, files);
      setSuccess(true);
      setTitles([]);
      setFiles([]);
      // Сбросить input
      const fileInput = document.getElementById('document-file');
      if (fileInput) fileInput.value = '';
    } catch (err) {
      setError(err.message || 'Ошибка при загрузке файлов');
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Загрузить новые документы
      </Typography>
      {error && <Alert severity="error" sx={{ mb: 2 }} data-testid="upload-error">{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>Документы успешно загружены!</Alert>}
      <Box
        component="form"
        onSubmit={handleSubmit}
        onDragEnter={handleDrag}
        sx={{ position: 'relative' }}
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
            cursor: 'pointer',
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
            disabled={loading}
          >
            Выбрать файлы
            <input
              id="document-file"
              type="file"
              accept={ACCEPT_STRING}
              hidden
              multiple
              onChange={handleFileChange}
              disabled={loading}
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
              disabled={loading}
            />
            <Box data-testid="source-select" sx={{ display: 'none' }} />
            <Button color="error" size="small" onClick={() => handleRemoveFile(idx)} sx={{ ml: 1 }}>
              Удалить
            </Button>
          </Box>
        ))}
        <Button
          type="submit"
          variant="contained"
          color="primary"
          fullWidth
          disabled={loading || !files.length}
          sx={{ mt: 2 }}
          startIcon={loading && <CircularProgress size={20} color="inherit" />}
          data-testid="submit-button"
        >
          {loading ? 'Загрузка...' : 'Загрузить документы'}
        </Button>
      </Box>
    </Paper>
  );
}

export default FileUpload;