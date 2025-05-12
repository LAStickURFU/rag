import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Slider,
  TextField,
  Button,
  Paper,
  Divider,
  Alert,
  IconButton,
  Tooltip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import { getModelSettings, updateModelSettings, getAvailableModels, switchModel } from '../services/api';

const ModelSettings = () => {
  const [settings, setSettings] = useState({
    temperature: 0.0,
    top_p: 0.9,
    max_tokens: 2048,
    top_k_chunks: 5,
    context_window: 8192
  });

  // Состояния для выбора модели (из ModelSelector)
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [switching, setSwitching] = useState(false);

  const [loading, setLoading] = useState(true);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [changed, setChanged] = useState(false);
  const [modelSuccess, setModelSuccess] = useState('');

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await getModelSettings();
        setSettings(data);
        setLoading(false);
      } catch (err) {
        setError('Не удалось загрузить настройки модели');
        setLoading(false);
      }
    };

    fetchSettings();

    // Загрузка списка доступных моделей
    const fetchModels = async () => {
      try {
        const data = await getAvailableModels();
        setModels(data.models);
        setActiveModel(data.active_model);
        setSelectedModel(data.active_model);
      } catch (err) {
        console.error('Ошибка при получении списка моделей:', err);
        setError('Не удалось загрузить список доступных моделей');
      }
    };

    fetchModels();
  }, []);

  const handleChange = (name) => (event, newValue) => {
    // Для слайдеров используем второй параметр
    const value = newValue !== undefined ? newValue : event.target.value;

    setSettings({
      ...settings,
      [name]: value
    });

    setChanged(true);
  };

  const handleNumberChange = (name) => (event) => {
    const value = parseFloat(event.target.value);

    if (!isNaN(value)) {
      setSettings({
        ...settings,
        [name]: value
      });

      setChanged(true);
    }
  };

  // Обработчик изменения выбранной модели
  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  // Обработчик кнопки смены модели
  const handleSwitchModel = async () => {
    if (selectedModel === activeModel) {
      return; // Модель уже активна
    }

    try {
      setError('');
      setModelSuccess('');
      setSwitching(true);

      await switchModel(selectedModel);

      setActiveModel(selectedModel);
      setModelSuccess(`Модель успешно изменена на ${selectedModel}`);
    } catch (err) {
      console.error('Ошибка при переключении модели:', err);
      setError('Не удалось переключить модель');
    } finally {
      setSwitching(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSuccess(false);
    setError('');

    try {
      setLoading(true);
      await updateModelSettings(settings);
      setSuccess(true);
      setChanged(false);
    } catch (err) {
      setError('Не удалось обновить настройки модели');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Accordion defaultExpanded={false}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="model-settings-content"
          id="model-settings-header"
        >
          <Typography variant="h6">Настройки генерации</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2 }}>Настройки успешно сохранены!</Alert>}
          {modelSuccess && <Alert severity="success" sx={{ mb: 2 }}>{modelSuccess}</Alert>}

          <Box component="form" onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              {/* Выбор языковой модели */}
              <Grid item xs={12}>
                <Typography variant="h6" sx={{ mb: 2 }}>Языковая модель</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body1">
                    Текущая модель: <strong>{activeModel}</strong>
                  </Typography>
                  <Tooltip title="Выберите языковую модель для генерации ответов. Учтите, что переключение требует загрузки новой модели и может занять некоторое время.">
                    <IconButton size="small" sx={{ ml: 1 }}>
                      <InfoIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={12} md={8}>
                    <FormControl fullWidth>
                      <InputLabel id="model-select-label">Выбрать модель</InputLabel>
                      <Select
                        labelId="model-select-label"
                        id="model-select"
                        value={selectedModel}
                        label="Выбрать модель"
                        onChange={handleModelChange}
                        disabled={switching}
                      >
                        {models.map((model) => (
                          <MenuItem key={model.name} value={model.name}>
                            <Box>
                              <Typography variant="body1">{model.name}</Typography>
                              {model.description && (
                                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                                  {model.description}
                                </Typography>
                              )}
                            </Box>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>

                  <Grid item xs={12} md={4}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleSwitchModel}
                      disabled={switching || selectedModel === activeModel}
                      fullWidth
                      sx={{ height: '56px' }} // Выравниваем по высоте с Select
                    >
                      {switching ? (
                        <>
                          <CircularProgress size={24} sx={{ mr: 1 }} />
                          Переключение...
                        </>
                      ) : (
                        'Переключить модель'
                      )}
                    </Button>
                  </Grid>
                </Grid>
                <Alert severity="info" sx={{ mt: 2, mb: 3 }}>
                  При переключении модели, новая модель будет загружена в память.
                  Это может занять некоторое время, особенно при первом запуске.
                </Alert>
                <Divider sx={{ my: 2 }} />
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mt: 2, mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography id="top-k-chunks-slider" gutterBottom>
                      Количество фрагментов для контекста
                    </Typography>
                    <Tooltip title="Определяет, сколько релевантных фрагментов документов будет включено в контекст для модели. Большее значение даёт более полную информацию, но может переполнить контекст.">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={settings.top_k_chunks}
                        onChange={handleChange('top_k_chunks')}
                        min={1}
                        max={20}
                        step={1}
                        valueLabelDisplay="auto"
                        aria-labelledby="top-k-chunks-slider"
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={settings.top_k_chunks}
                        onChange={handleNumberChange('top_k_chunks')}
                        inputProps={{
                          step: 1,
                          min: 1,
                          max: 20,
                          type: 'number',
                        }}
                        size="small"
                        sx={{ width: 60 }}
                      />
                    </Grid>
                  </Grid>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mt: 2, mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography id="temperature-slider" gutterBottom>
                      Температура
                    </Typography>
                    <Tooltip title="Контролирует креативность ответов. Низкие значения дают более предсказуемые ответы, высокие - более творческие.">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={settings.temperature}
                        onChange={handleChange('temperature')}
                        min={0}
                        max={1}
                        step={0.01}
                        valueLabelDisplay="auto"
                        aria-labelledby="temperature-slider"
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={settings.temperature}
                        onChange={handleNumberChange('temperature')}
                        inputProps={{
                          step: 0.01,
                          min: 0,
                          max: 1,
                          type: 'number',
                        }}
                        size="small"
                        sx={{ width: 60 }}
                      />
                    </Grid>
                  </Grid>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mt: 2, mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography id="top-p-slider" gutterBottom>
                      Top-P
                    </Typography>
                    <Tooltip title="Параметр nucleus sampling, ограничивающий выбор слов из наиболее вероятных. Значение 0.9 означает, что модель будет выбирать только из слов, суммарная вероятность которых составляет 90%.">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={settings.top_p}
                        onChange={handleChange('top_p')}
                        min={0.1}
                        max={1}
                        step={0.01}
                        valueLabelDisplay="auto"
                        aria-labelledby="top-p-slider"
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={settings.top_p}
                        onChange={handleNumberChange('top_p')}
                        inputProps={{
                          step: 0.01,
                          min: 0.1,
                          max: 1,
                          type: 'number',
                        }}
                        size="small"
                        sx={{ width: 60 }}
                      />
                    </Grid>
                  </Grid>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ mt: 2, mb: 3 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography id="max-tokens-slider" gutterBottom>
                      Максимальное количество токенов
                    </Typography>
                    <Tooltip title="Ограничивает длину ответа модели. Один токен примерно равен 4 символам для русского языка.">
                      <IconButton size="small">
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs>
                      <Slider
                        value={settings.max_tokens}
                        onChange={handleChange('max_tokens')}
                        min={100}
                        max={8192}
                        step={100}
                        valueLabelDisplay="auto"
                        aria-labelledby="max-tokens-slider"
                      />
                    </Grid>
                    <Grid item>
                      <TextField
                        value={settings.max_tokens}
                        onChange={handleNumberChange('max_tokens')}
                        inputProps={{
                          step: 100,
                          min: 100,
                          max: 8192,
                          type: 'number',
                        }}
                        size="small"
                        sx={{ width: 80 }}
                      />
                    </Grid>
                  </Grid>
                </Box>
              </Grid>

              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  disabled={loading || !changed}
                  sx={{ mt: 2 }}
                >
                  {loading ? 'Сохранение...' : 'Сохранить настройки'}
                </Button>
              </Grid>
            </Grid>
          </Box>
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
};

export default ModelSettings;