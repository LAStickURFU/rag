import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Paper, List, ListItem, ListItemText, Button, FormControl, 
         Select, MenuItem, Grid, Alert, Snackbar, CircularProgress, Divider, IconButton, Tooltip, Tabs, Tab } from '@mui/material';
import { updateUserRole, getAllUsers, deleteUser } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import AdminResetPasswordForm from '../components/AdminResetPasswordForm';
import DeleteIcon from '@mui/icons-material/Delete';
import SystemStats from '../components/SystemStats';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function DashboardPage() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });
  const { isAdmin, user } = useAuth();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    // Если пользователь не администратор, перенаправляем на главную
    if (user && !isAdmin()) {
      navigate('/');
    } else {
      fetchUsers();
    }
  }, [user, isAdmin, navigate]);

  const fetchUsers = async () => {
    setLoading(true);
    setError('');
    
    try {
      // Получаем список всех пользователей через API
      const usersList = await getAllUsers();
      setUsers(usersList);
    } catch (err) {
      setError('Не удалось загрузить список пользователей');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleRoleChange = async (username, newRole) => {
    try {
      await updateUserRole(username, newRole);
      
      // Обновляем локальное состояние
      setUsers(users.map(u => 
        u.username === username ? { ...u, role: newRole } : u
      ));
      
      setNotification({
        open: true,
        message: `Роль пользователя ${username} изменена на ${newRole}`,
        severity: 'success'
      });
    } catch (err) {
      setError(`Ошибка при изменении роли: ${err.message}`);
      setNotification({
        open: true,
        message: `Ошибка при изменении роли: ${err.message}`,
        severity: 'error'
      });
    }
  };

  const handlePasswordReset = () => {
    setNotification({
      open: true,
      message: 'Пароль пользователя успешно сброшен',
      severity: 'success'
    });
  };

  const handleDeleteUser = async (username) => {
    // Нельзя удалить себя
    if (username === user?.username) {
      setNotification({
        open: true,
        message: 'Невозможно удалить свою учетную запись',
        severity: 'error'
      });
      return;
    }

    if (!window.confirm(`Вы действительно хотите удалить пользователя ${username}? Это действие нельзя отменить.`)) {
      return;
    }

    try {
      await deleteUser(username);
      // Обновляем список пользователей после удаления
      await fetchUsers();
      setNotification({
        open: true,
        message: `Пользователь ${username} успешно удален`,
        severity: 'success'
      });
    } catch (err) {
      setError(`Ошибка при удалении пользователя: ${err.message || err.detail || 'Неизвестная ошибка'}`);
      setNotification({
        open: true,
        message: `Ошибка при удалении пользователя: ${err.message || err.detail || 'Неизвестная ошибка'}`,
        severity: 'error'
      });
    }
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  if (!isAdmin()) {
    return null; // Не показываем содержимое не администраторам
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Панель администратора
      </Typography>
      
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="Разделы панели администратора">
          <Tab label="Пользователи" />
          <Tab label="Статистика системы" />
        </Tabs>
      </Box>
      
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Управление пользователями
              </Typography>
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : users.length === 0 ? (
                <Typography variant="body1" color="textSecondary">
                  Пользователи не найдены
                </Typography>
              ) : (
                <List>
                  {users.map((userData) => (
                    <React.Fragment key={userData.id}>
                      <ListItem
                        secondaryAction={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 1 }}>
                              <Select
                                value={userData.role}
                                onChange={(e) => handleRoleChange(userData.username, e.target.value)}
                              >
                                <MenuItem value="user">Пользователь</MenuItem>
                                <MenuItem value="admin">Администратор</MenuItem>
                              </Select>
                            </FormControl>
                            
                            <Tooltip title="Удалить пользователя">
                              <IconButton 
                                edge="end" 
                                color="error" 
                                onClick={() => handleDeleteUser(userData.username)}
                                disabled={userData.username === user?.username}
                              >
                                <DeleteIcon />
                              </IconButton>
                            </Tooltip>
                          </Box>
                        }
                      >
                        <ListItemText
                          primary={userData.username}
                          secondary={`ID: ${userData.id}, Роль: ${userData.role}`}
                        />
                      </ListItem>
                      <Divider />
                    </React.Fragment>
                  ))}
                </List>
              )}
            </Paper>
            
            <Box sx={{ mt: 3 }}>
              <AdminResetPasswordForm users={users} onSuccess={handlePasswordReset} />
            </Box>
          </Grid>
          
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Системная информация
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>Текущий пользователь:</strong> {user ? user.username : 'Не авторизован'}
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>Роль:</strong> {user ? user.role : '-'}
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>ID пользователя:</strong> {user ? user.id : '-'}
              </Typography>
              
              <Typography variant="body1" paragraph>
                <strong>Дата регистрации:</strong> {user && user.created_at ? new Date(user.created_at).toLocaleString() : '-'}
              </Typography>
              
              <Box sx={{ mt: 3 }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => navigate('/documents')}
                  sx={{ mr: 2, mb: 1 }}
                >
                  Управление документами
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        <SystemStats />
      </TabPanel>
      
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        message={notification.message}
      />
    </Container>
  );
}

export default DashboardPage;