import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  CircularProgress,
  Alert,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  Grid
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { getEvaluationHistory, downloadEvaluationReport } from '../services/api';

const DATASETS = [
  { value: 'sberquad', label: 'SberQuAD' },
  { value: 'RuBQ', label: 'RuBQ' },
  { value: 'mUSER', label: 'mUSER' }
];

const METRIC_INFO = {
  faithfulness: {
    label: 'Faithfulness',
    description: 'Насколько ответ основан на retrieved контексте.'
  },
  answer_relevancy: {
    label: 'Answer Relevance',
    description: 'Насколько ответ релевантен вопросу.'
  },
  context_recall: {
    label: 'Context Recall',
    description: 'Вся ли нужная инфа попала в retrieved чанки.'
  },
  context_precision: {
    label: 'Context Precision',
    description: 'Доля релевантных чанков.'
  },
  context_relevance: {
    label: 'Context Relevance',
    description: 'Общая релевантность чанков вопросу.'
  },
  answer_correctness: {
    label: 'Answer Correctness',
    description: 'Совпадение с эталоном.'
  },
};

const getMetricColor = (value) => {
  if (value >= 0.8) return '#43a047';
  if (value >= 0.6) return '#ffa000';
  return '#e53935';
};

function EvaluationPage() {
  const [dataset, setDataset] = useState('sberquad');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [taskId, setTaskId] = useState(null);
  const [polling, setPolling] = useState(false);
  const logsEndRef = React.useRef(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [sampleSize, setSampleSize] = useState(5);
  const [loadingDatasetInfo, setLoadingDatasetInfo] = useState(false);
  
  // Модальное окно для детальной информации о примере
  const [detailModalOpen, setDetailModalOpen] = useState(false);
  const [selectedExample, setSelectedExample] = useState(null);

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const data = await getEvaluationHistory();
      setHistory(data);
    } catch (e) {
      setHistory([]);
    }
    setHistoryLoading(false);
  };

  React.useEffect(() => {
    fetchHistory();
  }, []);

  // Загрузка информации о выбранном датасете при его изменении
  React.useEffect(() => {
    const fetchDatasetInfo = async () => {
      setLoadingDatasetInfo(true);
      try {
        const response = await fetch(`/api/evaluation/datasets/${dataset}`);
        if (!response.ok) {
          throw new Error(`Не удалось загрузить информацию о датасете ${dataset}`);
        }
        const data = await response.json();
        setDatasetInfo(data);
        
        // Если количество элементов в датасете меньше текущего выбранного размера выборки,
        // корректируем размер выборки
        if (data.total_items < sampleSize) {
          setSampleSize(Math.min(data.total_items, 5));
        }
      } catch (e) {
        console.error("Ошибка при загрузке информации о датасете:", e);
        setDatasetInfo(null);
      } finally {
        setLoadingDatasetInfo(false);
      }
    };
    
    if (dataset) {
      fetchDatasetInfo();
    }
  }, [dataset, sampleSize]);

  // Polling for progress and logs
  React.useEffect(() => {
    if (!taskId || !polling) return;
    const interval = setInterval(async () => {
      try {
        const statusRes = await fetch(`/api/evaluate-rag/status/${taskId}`);
        const statusData = await statusRes.json();
        setProgress(statusData.progress || 0);
        if (statusData.status === 'done' || statusData.status === 'error') setPolling(false);
        const logsRes = await fetch(`/api/evaluate-rag/logs/${taskId}`);
        const logsData = await logsRes.json();
        setLogs(logsData.logs || []);
      } catch {}
    }, 1200);
    return () => clearInterval(interval);
  }, [taskId, polling]);

  React.useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const handleRunEvaluation = async () => {
    setLoading(true);
    setError('');
    setResults(null);
    setProgress(0);
    setLogs([]);
    setTaskId(null);
    try {
      // Используем информацию о датасете, которая уже загружена
      if (!datasetInfo || !datasetInfo.items || !datasetInfo.items.length) {
        throw new Error(`Датасет ${dataset} пуст или не загружен`);
      }
      
      // Проверяем размер выборки
      const actualSampleSize = Math.min(sampleSize, datasetInfo.items.length);
      if (actualSampleSize === 0) {
        throw new Error("Размер выборки для оценки должен быть больше 0");
      }
      
      // Логируем информацию о запуске оценки
      console.log(`Запуск оценки на ${actualSampleSize} из ${datasetInfo.total_items} элементов датасета ${dataset}`);
      
      // Новый асинхронный запуск
      const token = localStorage.getItem('token');
      const res = await fetch('/api/evaluate-rag/async', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        },
        body: JSON.stringify({ 
          eval_items: datasetInfo.items.slice(0, actualSampleSize).map(item => ({
            question: item.question,
            answer: item.answer || '',
            ground_truth: item.ground_truth
          })), 
          description: `${dataset} (${actualSampleSize} из ${datasetInfo.total_items})` 
        })
      });
      const data = await res.json();
      if (!data.task_id) throw new Error('Не удалось получить task_id');
      setTaskId(data.task_id);
      setPolling(true);
      // Ожидаем завершения
      const waitForDone = async () => {
        while (true) {
          const statusRes = await fetch(`/api/evaluate-rag/status/${data.task_id}`);
          const statusData = await statusRes.json();
          if (statusData.status === 'done') {
            // Получаем финальный результат
            const hist = await getEvaluationHistory();
            setHistory(hist);
            setLoading(false);
            setPolling(false);
            break;
          }
          if (statusData.status === 'error') {
            setError('Ошибка при запуске оценки: ' + (statusData.error || 'Неизвестная ошибка'));
            setLoading(false);
            setPolling(false);
            break;
          }
          await new Promise(r => setTimeout(r, 1200));
        }
      };
      waitForDone();
    } catch (err) {
      setError('Ошибка при запуске оценки: ' + (err.message || 'Неизвестная ошибка'));
      setLoading(false);
    }
  };

  const handleDownloadReport = async (filename) => {
    try {
      const blob = await downloadEvaluationReport(filename);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (e) {
      alert('Ошибка при скачивании отчёта');
    }
  };
  
  // Открытие модального окна с детальной информацией о примере
  const handleOpenExampleDetail = (example) => {
    setSelectedExample(example);
    setDetailModalOpen(true);
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>Оценка качества RAG</Typography>
        <Typography variant="body1" sx={{ mb: 2 }}>
          Выберите датасет для оценки качества работы вашей RAG-модели. Будут рассчитаны метрики RAGAS, а также сгенерировано краткое резюме. История запусков и подробные отчёты сохраняются на сервере.
        </Typography>
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="dataset-select-label">Датасет</InputLabel>
          <Select
            labelId="dataset-select-label"
            value={dataset}
            label="Датасет"
            onChange={e => setDataset(e.target.value)}
          >
            {DATASETS.map(ds => (
              <MenuItem key={ds.value} value={ds.value}>{ds.label}</MenuItem>
            ))}
          </Select>
        </FormControl>
        
        {/* Информация о датасете */}
        {loadingDatasetInfo ? (
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <CircularProgress size={20} sx={{ mr: 1 }} />
            <Typography variant="body2">Загрузка информации о датасете...</Typography>
          </Box>
        ) : datasetInfo ? (
          <Paper variant="outlined" sx={{ p: 2, mb: 3, bgcolor: '#f5f8ff' }}>
            <Typography variant="subtitle2" gutterBottom>Информация о датасете</Typography>
            <Typography variant="body2">
              Всего элементов: <b>{datasetInfo.total_items}</b><br />
              Поля: {datasetInfo.example_fields.join(', ')}
            </Typography>
            
            {/* Выбор размера выборки */}
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
              <Typography variant="body2" sx={{ mr: 2 }}>
                Размер выборки для оценки:
              </Typography>
              <Select
                value={sampleSize}
                size="small"
                onChange={(e) => setSampleSize(Number(e.target.value))}
                sx={{ minWidth: 100 }}
              >
                {[1, 3, 5, 10, 20, 50, 100].filter(n => n <= datasetInfo.total_items).map(n => (
                  <MenuItem key={n} value={n}>{n}</MenuItem>
                ))}
                <MenuItem key="all" value={datasetInfo.total_items}>Все ({datasetInfo.total_items})</MenuItem>
              </Select>
            </Box>
            {sampleSize > 10 && (
              <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
                Внимание: оценка большого количества примеров может занять длительное время.
              </Typography>
            )}
          </Paper>
        ) : error ? null : (
          <Alert severity="info" sx={{ mb: 3 }}>Выберите датасет для оценки</Alert>
        )}
        
        <Button
          variant="contained"
          color="primary"
          onClick={handleRunEvaluation}
          disabled={loading || polling || !datasetInfo}
          startIcon={loading || polling ? <CircularProgress size={20} color="inherit" /> : <RefreshIcon />}
          sx={{ mb: 3 }}
        >
          {(loading || polling) ? 'Оценка...' : 'Запустить оценку'}
        </Button>
        {(polling || progress > 0) && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress variant="determinate" value={progress} sx={{ height: 8, borderRadius: 2 }} />
            <Typography variant="body2" sx={{ mt: 1, fontFamily: 'monospace' }}>
              Прогресс: {progress}% {sampleSize > 0 && `(выбрано ${sampleSize} элементов)`}
            </Typography>
            <Paper variant="outlined" sx={{ mt: 2, p: 2, maxHeight: 180, overflowY: 'auto', bgcolor: '#f8fafc' }}>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>Логи процесса</Typography>
              {logs.length === 0 ? (
                <Typography variant="body2" color="text.disabled">Логи отсутствуют</Typography>
              ) : (
                logs.map((log, i) => (
                  <Typography key={i} variant="body2" sx={{ fontFamily: 'monospace' }}>{log}</Typography>
                ))
              )}
              <div ref={logsEndRef} />
            </Paper>
          </Box>
        )}
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        {results && (
          <Box mt={4}>
            <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6" gutterBottom>Результаты оценки</Typography>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <b>Датасет:</b> {results.dataset_name || dataset} <br/>
                <b>Дата:</b> {results.timestamp || '-'}
              </Typography>
              <TableContainer component={Paper} sx={{ mb: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Метрика</TableCell>
                      <TableCell align="right">Значение</TableCell>
                      <TableCell>Описание</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(results.metrics).map(([key, value]) => (
                      <TableRow key={key}>
                        <TableCell>{METRIC_INFO[key]?.label || key}</TableCell>
                        <TableCell align="right">
                          <span style={{
                            fontWeight: 600,
                            color: getMetricColor(value),
                            background: getMetricColor(value) + '22',
                            borderRadius: 4,
                            padding: '2px 8px',
                          }}>{value.toFixed(4)}</span>
                        </TableCell>
                        <TableCell style={{ fontSize: 13 }}>
                          {METRIC_INFO[key]?.description || '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <Typography variant="h6" gutterBottom>Резюме</Typography>
              <Card variant="outlined" sx={{ mb: 2 }}>
                <CardContent>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>{results.summary}</Typography>
                </CardContent>
              </Card>
              
              {/* Детальный анализ примеров */}
              {results.example_reports && results.example_reports.length > 0 && (
                <>
                  <Typography variant="h6" gutterBottom>Детальный анализ примеров</Typography>
                  <Box sx={{ mb: 2 }}>
                    <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
                      <Table size="small" stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>#</TableCell>
                            <TableCell>Вопрос</TableCell>
                            <TableCell>Ответ модели</TableCell>
                            <TableCell>Эталонный ответ</TableCell>
                            <TableCell>Время поиска (сек)</TableCell>
                            <TableCell>Время генерации (сек)</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {results.example_reports.map((example, idx) => (
                            <TableRow 
                              key={idx} 
                              sx={{ 
                                '&:nth-of-type(odd)': { bgcolor: 'rgba(0, 0, 0, 0.02)' },
                                cursor: 'pointer',
                                '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.05)' }
                              }}
                              onClick={() => handleOpenExampleDetail(example)}
                            >
                              <TableCell>{idx + 1}</TableCell>
                              <TableCell sx={{ maxWidth: 200, overflowWrap: 'break-word' }}>{example.question}</TableCell>
                              <TableCell sx={{ maxWidth: 200, overflowWrap: 'break-word' }}>{example.response}</TableCell>
                              <TableCell sx={{ maxWidth: 200, overflowWrap: 'break-word' }}>{example.ground_truth}</TableCell>
                              <TableCell>{example.search_time_sec?.toFixed(2) || '-'}</TableCell>
                              <TableCell>{example.generation_time_sec?.toFixed(2) || '-'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                </>
              )}
            </Paper>
          </Box>
        )}
        <Box mt={6}>
          <Typography variant="h5" gutterBottom>История запусков оценки</Typography>
          {historyLoading ? <CircularProgress /> : (
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Дата</TableCell>
                    <TableCell>Датасет</TableCell>
                    <TableCell>Метрики</TableCell>
                    <TableCell>Summary</TableCell>
                    <TableCell>Отчёт</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.map((item, idx) => (
                    <TableRow key={item.filename || idx}>
                      <TableCell>{item.timestamp || '-'}</TableCell>
                      <TableCell>{item.dataset_name || '-'}</TableCell>
                      <TableCell>
                        {item.metrics ? Object.entries(item.metrics).map(([k, v]) => `${k}: ${v.toFixed(3)}`).join(', ') : '-'}
                      </TableCell>
                      <TableCell style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.summary ? item.summary.slice(0, 80) + (item.summary.length > 80 ? '…' : '') : '-'}
                      </TableCell>
                      <TableCell>
                        {item.filename && (
                          <Button size="small" variant="outlined" onClick={() => handleDownloadReport(item.filename)}>
                            Скачать
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Box>
      </Paper>
      
      {/* Модальное окно с детальной информацией о примере */}
      {selectedExample && (
        <Dialog 
          open={detailModalOpen} 
          onClose={() => setDetailModalOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            Детальная информация о примере
          </DialogTitle>
          <DialogContent dividers>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>Вопрос:</Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f8f9fa', mb: 2 }}>
                  <Typography>{selectedExample.question}</Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>Ответ модели:</Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f0f7ff', mb: 2 }}>
                  <Typography whiteSpace="pre-wrap">{selectedExample.response}</Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>Эталонный ответ:</Typography>
                <Paper variant="outlined" sx={{ p: 2, bgcolor: '#f0fff0', mb: 2 }}>
                  <Typography whiteSpace="pre-wrap">{selectedExample.ground_truth}</Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" gutterBottom>Найденные контексты:</Typography>
                {selectedExample.retrieved_contexts && selectedExample.retrieved_contexts.length > 0 ? (
                  selectedExample.retrieved_contexts.map((context, i) => (
                    <Paper 
                      key={i} 
                      variant="outlined" 
                      sx={{ p: 2, mb: 1, bgcolor: '#f5f5f5' }}
                    >
                      <Typography variant="caption" display="block" gutterBottom>
                        Контекст #{i+1} {selectedExample.context_scores && 
                          `(релевантность: ${selectedExample.context_scores[i]?.toFixed(4) || 'н/д'})`}
                      </Typography>
                      <Typography variant="body2">{context}</Typography>
                    </Paper>
                  ))
                ) : (
                  <Typography color="text.secondary">Контексты не найдены</Typography>
                )}
              </Grid>
              
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>Статистика:</Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                  <Typography variant="body2">
                    Время поиска: <b>{selectedExample.search_time_sec?.toFixed(2) || 'н/д'} сек</b>
                  </Typography>
                  <Typography variant="body2">
                    Время генерации: <b>{selectedExample.generation_time_sec?.toFixed(2) || 'н/д'} сек</b>
                  </Typography>
                  <Typography variant="body2">
                    Найдено контекстов: <b>{selectedExample.retrieved_contexts?.length || 0}</b>
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDetailModalOpen(false)}>Закрыть</Button>
          </DialogActions>
        </Dialog>
      )}
    </Container>
  );
}

export default EvaluationPage;