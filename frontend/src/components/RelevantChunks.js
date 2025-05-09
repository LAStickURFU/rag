import React from 'react';
import {
  Box,
  Typography,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper,
  Alert,
  Link
} from '@mui/material';
import ArticleIcon from '@mui/icons-material/Article';

const RelevantChunks = ({ chunks = [] }) => {
  // Если нет чанков, отображаем сообщение
  const hasChunks = chunks && chunks.length > 0;

  // Сортируем чанки по релевантности (если они есть)
  const sortedChunks = hasChunks
    ? [...chunks].sort((a, b) => a.relevance - b.relevance)
    : [];

  return (
    <Box>
      {!hasChunks ? (
        <Alert severity="info" sx={{ mb: 1 }}>
          <Box component="span" sx={{ display: 'block' }}>
            Система не использовала никаких документов для формирования ответа.
            Это может быть связано с отсутствием подходящей информации в базе знаний.
          </Box>
          <Box component="span" sx={{ display: 'block', mt: 1 }}>
            Попробуйте <Link href="/documents" underline="hover">загрузить документы</Link> или
            <Link href="#" underline="hover" onClick={(e) => { e.preventDefault(); window.location.reload(); }}>
              {' переиндексировать существующие'}
            </Link>.
          </Box>
        </Alert>
      ) : (
        <List disablePadding>
          {sortedChunks.map((chunk, index) => (
            <React.Fragment key={`chunk-${index}`}>
              <ListItem
                alignItems="flex-start"
                sx={{
                  bgcolor: index % 2 === 0 ? 'background.paper' : '#f9f9f9',
                  px: 2,
                  py: 1,
                  '&:hover': { bgcolor: '#edf2f7' }
                }}
                data-testid="message-chunk"
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <ArticleIcon fontSize="small" color="primary" sx={{ mr: 1 }} />
                        <Typography variant="subtitle2" color="primary" component="div">
                          {chunk.metadata?.title
                            ? `${chunk.metadata.title}`
                            : `Фрагмент #${index + 1}`}
                        </Typography>
                        {chunk.metadata?.source && (
                          <Chip
                            label={chunk.metadata.source}
                            size="small"
                            variant="outlined"
                            sx={{ ml: 1, fontSize: '0.65rem' }}
                          />
                        )}
                      </Box>
                      <Chip
                        label={`Релевантность: ${(1 - chunk.relevance).toFixed(2)}`}
                        size="small"
                        color={index < 3 ? "success" : "default"}
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Paper elevation={0} variant="outlined" sx={{ mt: 1, p: 1 }}>
                      <Box
                        component="span"
                        sx={{
                          whiteSpace: 'pre-wrap',
                          maxHeight: '150px',
                          overflow: 'auto',
                          fontFamily: 'monospace',
                          fontSize: '0.85rem',
                          lineHeight: 1.5
                        }}
                      >
                        {chunk.text}
                      </Box>
                    </Paper>
                  }
                  secondaryTypographyProps={{ component: 'div' }}
                />
              </ListItem>
              {index < sortedChunks.length - 1 && <Divider component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Box>
  );
};

export default RelevantChunks;