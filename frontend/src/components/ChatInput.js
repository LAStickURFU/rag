import React, { useState } from 'react';
import {
  Paper,
  InputBase,
  IconButton,
  CircularProgress,
  FormControlLabel,
  Switch,
  Typography
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

const ChatInput = ({ onSendMessage, loading, disabled }) => {
  const [message, setMessage] = useState('');
  const [useDirect, setUseDirect] = useState(false); // По умолчанию используем RAG

  const handleSend = () => {
    if (message.trim()) {
      onSendMessage(message.trim(), useDirect);
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <FormControlLabel
        control={
          <Switch
            checked={useDirect}
            onChange={(e) => setUseDirect(e.target.checked)}
            color="primary"
          />
        }
        label={
          <Typography variant="body2" color="textSecondary">
            {useDirect ? "Прямой запрос к LLM" : "Запрос с использованием RAG"}
          </Typography>
        }
        sx={{ mb: 1 }}
      />
      <Paper
        component="form"
        sx={{
          p: '2px 4px',
          display: 'flex',
          alignItems: 'center',
          borderRadius: 2,
          boxShadow: 2,
          mb: 0
        }}
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <InputBase
          sx={{ ml: 1, flex: 1 }}
          placeholder="Введите ваш вопрос..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          multiline
          minRows={4}
          maxRows={10}
          disabled={loading || disabled}
        />
        {loading ? (
          <>
            <CircularProgress size={24} sx={{ mx: 1 }} role="progressbar" />
            <button disabled style={{ display: 'none' }}>Отправка...</button>
            <button disabled style={{ display: 'none' }}>Отправить</button>
          </>
        ) : (
          <>
            <IconButton
              sx={{ p: '10px', color: 'primary.main' }}
              onClick={handleSend}
              disabled={!message.trim() || disabled}
            >
              <SendIcon />
            </IconButton>
            <button disabled={loading || !message.trim() || disabled} style={{ display: 'none' }}>Отправить</button>
          </>
        )}
      </Paper>
    </>
  );
};

export default ChatInput;