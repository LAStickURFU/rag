import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Avatar,
  Chip
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import RelevantChunks from './RelevantChunks';

const MessageItem = ({ message, isUser, relevantChunks, answerMode }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        mb: 2
      }}
      data-testid={isUser ? 'user-message' : 'bot-message'}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          maxWidth: '80%',
        }}
      >
        {!isUser && (
          <Avatar
            sx={{
              bgcolor: 'primary.main',
              mr: 1,
              width: 32,
              height: 32
            }}
          >
            <SmartToyIcon fontSize="small" />
          </Avatar>
        )}
      
        <Box sx={{ position: 'relative', width: '100%' }}>
          <Paper
            elevation={1}
            sx={{
              p: 2,
              bgcolor: isUser ? 'primary.light' : 'background.paper',
              borderRadius: 2,
              color: isUser ? 'white' : 'text.primary',
            }}
          >
            <Typography sx={{ whiteSpace: 'pre-wrap' }} component="div">
              {typeof message === 'string' ? message : String(message?.text)}
            </Typography>
            {/* Бейдж режима ответа теперь внутри Paper, под текстом */}
            {!isUser && answerMode && (
              <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
                <Chip
                  label={answerMode === 'rag' ? 'RAG' : 'LLM'}
                  color={answerMode === 'rag' ? 'success' : 'info'}
                  size="small"
                  sx={{ fontWeight: 600 }}
                  data-testid="answer-mode-chip"
                />
              </Box>
            )}
          </Paper>
        </Box>
      
        {isUser && (
          <Avatar
            sx={{
              bgcolor: 'secondary.main',
              ml: 1,
              width: 32,
              height: 32
            }}
          >
            <PersonIcon fontSize="small" />
          </Avatar>
        )}
      </Box>
      {!isUser && relevantChunks && relevantChunks.length > 0 && (
        <RelevantChunks chunks={relevantChunks} />
      )}
    </Box>
  );
};

export default MessageItem;