import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  Link,
  CircularProgress 
} from '@mui/material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

function RegisterPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [error, setError] = useState('');
  const { register, login, loading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
   
    if (!username.trim()) {
      setError('Введите имя пользователя');
      return;
    }
   
    if (!password.trim()) {
      setError('Введите пароль');
      return;
    }
   
    if (password !== passwordConfirm) {
      setError('Пароли не совпадают');
      return;
    }
   
    if (password.length < 6) {
      setError('Пароль должен содержать не менее 6 символов');
      return;
    }
   
    try {
      // Регистрируем пользователя
      await register(username, password);
     
      // Автоматически авторизуем и перенаправляем на чат
      await login(username, password);
      navigate('/chat');
    } catch (err) {
      setError(err.message || 'Ошибка при регистрации');
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" align="center" gutterBottom>
          Регистрация
        </Typography>
       
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
       
        <Box component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Имя пользователя"
            variant="outlined"
            margin="normal"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
            autoFocus
            InputLabelProps={{ shrink: true }}
          />
         
          <TextField
            fullWidth
            label="Пароль"
            type="password"
            variant="outlined"
            margin="normal"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            InputLabelProps={{ shrink: true }}
          />
         
          <TextField
            fullWidth
            label="Подтверждение пароля"
            type="password"
            variant="outlined"
            margin="normal"
            value={passwordConfirm}
            onChange={(e) => setPasswordConfirm(e.target.value)}
            disabled={loading}
            InputLabelProps={{ shrink: true }}
          />
         
          <Button
            type="submit"
            fullWidth
            variant="contained"
            color="primary"
            size="large"
            disabled={loading}
            sx={{ mt: 3, mb: 2 }}
            startIcon={loading && <CircularProgress size={20} color="inherit" />}
          >
            {loading ? 'Регистрация...' : 'Зарегистрироваться'}
          </Button>
         
          <Box sx={{ textAlign: 'center', mt: 2 }}>
            <Typography variant="body2">
              Уже есть аккаунт?{' '}
              <Link component={RouterLink} to="/login">
                Войти
              </Link>
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
}

export default RegisterPage;