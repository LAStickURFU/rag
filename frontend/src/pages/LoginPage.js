import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  TextField,
  Button,
  Typography,
  Box,
  Paper,
  CircularProgress,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import { getToken, registerUser, isAuthenticated } from '../services/api';

function LoginPage({ loading: loadingProp, setLoading: setLoadingProp }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [internalLoading, internalSetLoading] = useState(false);
  const loading = loadingProp !== undefined ? loadingProp : internalLoading;
  const setLoading = setLoadingProp !== undefined ? setLoadingProp : internalSetLoading;
  const [error, setError] = useState('');
  const [tab, setTab] = useState(0); // 0 - логин, 1 - регистрация
 
  const navigate = useNavigate();

  // Редирект если уже залогинен
  useEffect(() => {
    if (isAuthenticated()) {
      navigate('/chat');
    }
  }, [navigate]);

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
    setError('');
  };

  const handleLogin = async (e) => {
    e.preventDefault();
   
    if (!username || !password) {
      setError('Заполните все поля');
      return;
    }
   
    setLoading(true);
    setError('');
   
    try {
      const response = await getToken(username, password);
     
      // Сохраняем токен
      localStorage.setItem('token', response.access_token);
     
      // Переходим на страницу чата
      window.location.replace('/chat');
    } catch (error) {
      console.error('Error during login:', error);
      setError(error.message || 'Ошибка при входе. Проверьте логин и пароль.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
   
    if (!username || !password) {
      setError('Заполните все поля');
      return;
    }
   
    if (username.length < 3) {
      setError('Имя пользователя должно содержать не менее 3 символов');
      return;
    }
   
    if (password.length < 6) {
      setError('Пароль должен содержать не менее 6 символов');
      return;
    }
   
    setLoading(true);
    setError('');
   
    try {
      await registerUser(username, password);
     
      // После успешной регистрации автоматически логинимся
      const response = await getToken(username, password);
      localStorage.setItem('token', response.access_token);
     
      window.location.replace('/chat');
    } catch (error) {
      console.error('Error during registration:', error);
      setError(error.message || 'Ошибка при регистрации. Возможно, пользователь уже существует.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" align="center" gutterBottom>
          RAG Сервис
        </Typography>
       
        <Tabs value={tab} onChange={handleTabChange} centered sx={{ mb: 4 }}>
          <Tab label="Вход" />
          <Tab label="Регистрация" />
        </Tabs>
       
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
       
        <Box component="form" onSubmit={tab === 0 ? handleLogin : handleRegister}>
          <TextField
            label="Имя пользователя"
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            fullWidth
            margin="normal"
            variant="outlined"
            required
            disabled={loading}
            autoFocus
            InputLabelProps={{ shrink: true }}
            data-testid="username-input"
          />
         
          <TextField
            label="Пароль"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
            margin="normal"
            variant="outlined"
            required
            disabled={loading}
            InputLabelProps={{ shrink: true }}
            data-testid="password-input"
          />
         
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            size="large"
            disabled={loading}
            sx={{ mt: 3, mb: 2 }}
            data-testid="login-button"
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : tab === 0 ? 'Войти' : 'Зарегистрироваться'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default LoginPage;