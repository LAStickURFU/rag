import React from 'react';
import { Container, Typography, Box, Paper, Grid, Divider } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import ChangePasswordForm from '../components/ChangePasswordForm';

function ProfilePage() {
  const { user } = useAuth();

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Профиль пользователя
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Информация о пользователе
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="body1" gutterBottom>
                <strong>Имя пользователя:</strong> {user?.username || '-'}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>Роль:</strong> {user?.role === 'admin' ? 'Администратор' : 'Пользователь'}
              </Typography>
              
              <Typography variant="body1" gutterBottom>
                <strong>Дата регистрации:</strong> {user?.created_at 
                  ? new Date(user.created_at).toLocaleDateString('ru-RU', {
                      year: 'numeric',
                      month: 'long', 
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    }) 
                  : '-'}
              </Typography>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="body2" color="text.secondary">
              Для изменения данных профиля обратитесь к администратору.
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <ChangePasswordForm />
        </Grid>
      </Grid>
    </Container>
  );
}

export default ProfilePage; 