import React, { useState } from 'react';
import { Box, TextField, Button, Alert, Typography, Paper, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { resetUserPassword } from '../services/api';

const AdminResetPasswordForm = ({ users = [], onSuccess }) => {
  const [selectedUsername, setSelectedUsername] = useState('');
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
    if (!selectedUsername || !newPassword || !confirmPassword) {
      setError('Пожалуйста, заполните все поля');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Новый пароль и подтверждение не совпадают');
      return;
    }

    setLoading(true);
    try {
      await resetUserPassword(selectedUsername, newPassword);
      setSuccess(`Пароль пользователя ${selectedUsername} успешно изменен`);
      // Очищаем поля
      setNewPassword('');
      setConfirmPassword('');
      // Вызываем колбэк успешного сброса, если он передан
      if (onSuccess) {
        onSuccess();
      }
    } catch (err) {
      setError(err.detail || err.message || 'Произошла ошибка при сбросе пароля');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Сброс пароля пользователя
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
      
      <Box component="form" onSubmit={handleSubmit}>
        <FormControl fullWidth margin="normal" required>
          <InputLabel id="user-select-label">Пользователь</InputLabel>
          <Select
            labelId="user-select-label"
            value={selectedUsername}
            label="Пользователь"
            onChange={(e) => setSelectedUsername(e.target.value)}
          >
            {users.map((user) => (
              <MenuItem key={user.id} value={user.username}>
                {user.username}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
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
          disabled={loading || !selectedUsername}
        >
          {loading ? 'Сброс пароля...' : 'Сбросить пароль'}
        </Button>
      </Box>
    </Paper>
  );
};

export default AdminResetPasswordForm; 