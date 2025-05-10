import React, { useState } from 'react';
import { Box, TextField, Button, Alert, Typography, Paper } from '@mui/material';
import { changePassword } from '../services/api';

const ChangePasswordForm = ({ onSuccess }) => {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Валидация
    if (!currentPassword || !newPassword || !confirmPassword) {
      setError('Пожалуйста, заполните все поля');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Новый пароль и подтверждение не совпадают');
      return;
    }

    if (newPassword === currentPassword) {
      setError('Новый пароль должен отличаться от текущего');
      return;
    }

    setLoading(true);
    try {
      await changePassword(currentPassword, newPassword);
      setSuccess('Пароль успешно изменен');
      // Очищаем поля
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      // Вызываем колбэк успешного изменения, если он передан
      if (onSuccess) {
        onSuccess();
      }
    } catch (err) {
      setError(err.detail || err.message || 'Произошла ошибка при изменении пароля');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Изменение пароля
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
      
      <Box component="form" onSubmit={handleSubmit}>
        <TextField
          label="Текущий пароль"
          type="password"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <TextField
          label="Новый пароль"
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <TextField
          label="Подтвердите новый пароль"
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <Button 
          type="submit" 
          variant="contained" 
          color="primary" 
          sx={{ mt: 2 }}
          disabled={loading}
        >
          {loading ? 'Сохранение...' : 'Изменить пароль'}
        </Button>
      </Box>
    </Paper>
  );
};

export default ChangePasswordForm; 