import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InfoIcon from '@mui/icons-material/Info';
import { getAvailableModels, switchModel } from '../services/api';

const ModelSelector = () => {
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Загрузка списка доступных моделей при монтировании компонента
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const data = await getAvailableModels();
        setModels(data.models);
        setActiveModel(data.active_model);
        setSelectedModel(data.active_model);
        setLoading(false);
      } catch (err) {
        console.error('Ошибка при получении списка моделей:', err);
        setError('Не удалось загрузить список доступных моделей');
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

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
      setSuccess('');
      setSwitching(true);
      
      await switchModel(selectedModel);
      
      setActiveModel(selectedModel);
      setSuccess(`Модель успешно изменена на ${selectedModel}`);
    } catch (err) {
      console.error('Ошибка при переключении модели:', err);
      setError('Не удалось переключить модель');
    } finally {
      setSwitching(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Accordion defaultExpanded={true}>
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="model-selector-content"
          id="model-selector-header"
        >
          <Typography variant="h6">Языковая модель</Typography>
        </AccordionSummary>
        <AccordionDetails>
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
          
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress />
            </Box>
          ) : (
            <Box>
              <Grid container spacing={2}>
                <Grid item xs={12}>
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
                </Grid>
                
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
                
                <Grid item xs={12}>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    При переключении модели, новая модель будет загружена в память. 
                    Это может занять некоторое время, особенно при первом запуске.
                  </Alert>
                </Grid>
              </Grid>
            </Box>
          )}
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
};

export default ModelSelector; 