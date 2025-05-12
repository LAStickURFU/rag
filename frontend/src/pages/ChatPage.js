import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Grid,
  Divider
} from '@mui/material';
import { askQuestion, getChatHistory, askDirectQuestion, clearChatHistory } from '../services/api';
import ChatInput from '../components/ChatInput';
import MessageItem from '../components/MessageItem';
import ModelSettings from '../components/ModelSettings';
import RelevantChunks from '../components/RelevantChunks';
import DeleteIcon from '@mui/icons-material/Delete';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import MetaInfoBlock from '../components/MetaInfoBlock';

function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [initialLoading, setInitialLoading] = useState(true);
  const [selectedChunks, setSelectedChunks] = useState([]);
  const [lastMeta, setLastMeta] = useState(null);

  const messagesEndRef = useRef(null);
  const chatPaperRef = useRef(null);

  // Загрузка истории чата при монтировании компонента
  useEffect(() => {
    loadChatHistory();
  }, []);

  // Прокрутка к последнему сообщению при добавлении нового
  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Загрузка истории чата
  const loadChatHistory = async () => {
    try {
      const history = await getChatHistory();
      if (history && history.length > 0) {
        // Сортируем по времени (старые сверху, новые снизу)
        const sortedHistory = [...history].sort((a, b) => new Date(a.created_at || a.timestamp) - new Date(b.created_at || b.timestamp));
        // Преобразование истории к формату сообщений
        const chatMessages = sortedHistory.map(chat => ([
          { text: chat.question, isUser: true },
          {
            text: chat.response,
            isUser: false,
            relevantChunks: chat.relevant_chunks || [],
            answerMode: chat.rag_used === false ? 'direct' : 'rag',
            rag_used: typeof chat.rag_used === 'boolean' ? chat.rag_used : true
          }
        ])).flat();
        setMessages(chatMessages);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
      setError('Не удалось загрузить историю чата');
    } finally {
      setInitialLoading(false);
    }
  };

  // Обработка отправки сообщения
  const handleSendMessage = async (text, useDirect = false) => {
    if (!text.trim()) return;

    // Добавляем сообщение пользователя
    const userMessage = { text, isUser: true };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    // Сбрасываем выбранные чанки при новом вопросе
    setSelectedChunks([]);

    setLoading(true);
    setError('');

    try {
      // Выбираем API в зависимости от режима
      const apiMethod = useDirect ? askDirectQuestion : askQuestion;
      const response = await apiMethod(text);

      // Добавляем ответ системы
      const botMessage = {
        text: response.response,
        isUser: false,
        relevantChunks: response.relevant_chunks || [],
        answerMode: response.rag_used === false ? 'direct' : 'rag',
        rag_used: typeof response.rag_used === 'boolean' ? response.rag_used : true
      };

      setMessages(prevMessages => [...prevMessages, botMessage]);

      // Если есть релевантные чанки, отображаем их
      if (response.relevant_chunks && response.relevant_chunks.length > 0) {
        setSelectedChunks(response.relevant_chunks);
      }
      // Сохраняем метаинформацию для debug-блока
      setLastMeta(response.meta || null);
    } catch (error) {
      console.error('Error sending message:', error);
      setError('Не удалось получить ответ');
    } finally {
      setLoading(false);
    }
  };

  // Очистка истории чата
  const handleClearChat = async () => {
    if (!window.confirm('Очистить всю историю чата?')) return;
    try {
      await clearChatHistory();
      setMessages([]);
    } catch (err) {
      setError(err.message || 'Ошибка при очистке истории чата');
    }
  };

  if (initialLoading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh'
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box
      sx={{
        height: 'calc(100vh - 64px)', // Увеличиваем отступ, учитывая высоту шапки (примерно 48-64px)
        display: 'flex',
        flexDirection: 'column',
        pt: 2,
        pb: 1,
        overflow: 'hidden' // Скрываем внешний скролл страницы
      }}
    >
      <Container
        maxWidth="xl"
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}
      >
        <Grid
          container
          spacing={1}
          sx={{
            height: '100%',
            flexWrap: 'nowrap' // Важно, чтобы Grid не переносился
          }}
        >
          {/* Левая часть - чат */}
          <Grid
            item
            xs={7}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              minWidth: 0 // Важно для правильного расчета flexbox 
            }}
          >
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                height: '100%'
              }}
            >
              {/* Блок чата */}
              <Paper
                ref={chatPaperRef}
                sx={{
                  flex: 1,
                  mb: 2,
                  p: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden', // Скрываем внешний скролл
                  height: 'calc(100% - 175px)' // Увеличиваем резерв под настройки и инпут
                }}
              >
                {/* Шапка чата */}
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    px: 2,
                    py: 0.5,
                    borderBottom: '1px solid #eee'
                  }}
                >
                  <Typography variant="h6" sx={{ fontWeight: 600 }}>
                    Чат с RAG-системой
                  </Typography>
                  <Tooltip title="Очистить чат">
                    <IconButton onClick={handleClearChat} size="small" sx={{ color: '#e53935' }}>
                      <DeleteIcon />
                    </IconButton>
                  </Tooltip>
                </Box>

                {/* История сообщений */}
                <Box
                  sx={{
                    flex: 1,
                    p: 2,
                    overflowY: 'auto',
                    display: 'flex',
                    flexDirection: 'column'
                  }}
                >
                  {messages.length === 0 ? (
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        height: '100%',
                        flexDirection: 'column',
                        color: 'text.secondary'
                      }}
                    >
                      <Typography variant="h6" gutterBottom>
                        Начните диалог с системой
                      </Typography>
                      <Typography variant="body2">
                        Задайте вопрос, чтобы начать общение
                      </Typography>
                    </Box>
                  ) : (
                    messages.map((message, index) => (
                      <MessageItem
                        key={index}
                        message={message}
                        isUser={message.isUser}
                        answerMode={message.answerMode}
                        ragUsed={message.rag_used}
                      />
                    ))
                  )}
                  <div ref={messagesEndRef} />
                </Box>
              </Paper>

              {/* Блок ошибок */}
              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {/* Блок настроек и ввода */}
              <Box sx={{ mb: 0 }}>
                <ModelSettings />
                <ChatInput
                  onSendMessage={handleSendMessage}
                  loading={loading}
                />
              </Box>
            </Box>
          </Grid>

          {/* Правая часть - релевантные чанки + метаинфо */}
          <Grid
            item
            xs={5}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              minWidth: 0 // Важно для правильного расчета flexbox
            }}
          >
            <Paper
              sx={{
                p: 1.5,
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                overflow: 'hidden'
              }}
            >
              <Typography variant="h6" gutterBottom>
                Релевантные фрагменты
              </Typography>
              <Box sx={{ flex: 1, overflowY: 'auto' }}>
                <RelevantChunks chunks={selectedChunks} data-testid="message-chunks" />
                <Divider sx={{ my: 3 }} />
                <MetaInfoBlock meta={lastMeta} />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default ChatPage;