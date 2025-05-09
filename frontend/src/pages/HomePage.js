import React from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  Paper,
  Grid,
  Card,
  CardContent,
  CardActions
} from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useAuth } from '../contexts/AuthContext';

function HomePage() {
  const { isAuthenticated } = useAuth();

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      {/* Главный баннер */}
      <Paper
        elevation={3}
        sx={{
          p: { xs: 3, md: 6 },
          mb: 4,
          background: 'linear-gradient(45deg, #673ab7 30%, #3f51b5 90%)',
          color: 'white'
        }}
      >
        <Typography variant="h3" component="h1" gutterBottom>
          RAG-сервис для работы с документами
        </Typography>
        <Typography variant="h6" paragraph>
          Задавайте вопросы по вашим документам и получайте точные ответы с использованием
          технологии Retrieval-Augmented Generation (RAG)
        </Typography>
        <Box sx={{ mt: 3 }}>
          {isAuthenticated ? (
            <Button
              variant="contained"
              color="secondary"
              size="large"
              component={RouterLink}
              to="/chat"
              endIcon={<ChatIcon />}
            >
              Начать общение
            </Button>
          ) : (
            <Button
              variant="contained"
              color="secondary"
              size="large"
              component={RouterLink}
              to="/register"
            >
              Зарегистрироваться
            </Button>
          )}
        </Box>
      </Paper>

      {/* Карточки возможностей */}
      <Typography variant="h4" component="h2" gutterBottom>
        Возможности системы
      </Typography>
      <Grid container spacing={4} sx={{ mb: 6 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h3" gutterBottom>
                Загрузка документов
              </Typography>
              <Typography variant="body1">
                Загружайте текстовые документы разных форматов, которые будут автоматически
                проиндексированы для быстрого поиска информации.
              </Typography>
            </CardContent>
            <CardActions>
              {isAuthenticated && (
                <Button
                  component={RouterLink}
                  to="/documents"
                  endIcon={<CloudUploadIcon />}
                >
                  Управление документами
                </Button>
              )}
            </CardActions>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h3" gutterBottom>
                Вопросно-ответное взаимодействие
              </Typography>
              <Typography variant="body1">
                Задавайте любые вопросы по загруженным документам и получайте точные
                ответы, основанные на содержимом ваших файлов.
              </Typography>
            </CardContent>
            <CardActions>
              {isAuthenticated && (
                <Button
                  component={RouterLink}
                  to="/chat"
                  endIcon={<ChatIcon />}
                >
                  Открыть чат
                </Button>
              )}
            </CardActions>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h5" component="h3" gutterBottom>
                История общения
              </Typography>
              <Typography variant="body1">
                Все ваши вопросы и ответы сохраняются в истории для последующего просмотра
                и анализа. Вы всегда можете вернуться к предыдущим запросам.
              </Typography>
            </CardContent>
            <CardActions>
              {isAuthenticated && (
                <Button
                  component={RouterLink}
                  to="/chat"
                >
                  Просмотреть историю
                </Button>
              )}
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* О технологии RAG */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h4" component="h2" gutterBottom>
          О технологии RAG
        </Typography>
        <Typography variant="body1" paragraph>
          Retrieval-Augmented Generation (RAG) — это подход, который объединяет в себе возможности
          поисковых систем и генеративных языковых моделей. RAG позволяет находить релевантную
          информацию в ваших документах и использовать её для генерации точных ответов на запросы.
        </Typography>
        <Typography variant="body1">
          Благодаря этой технологии система может дать подробный и точный ответ на основе
          именно ваших документов, без необходимости обучать модель на ваших данных.
        </Typography>
      </Paper>
    </Container>
  );
}

export default HomePage;