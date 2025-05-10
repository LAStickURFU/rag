import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  useMediaQuery,
  IconButton,
  Menu,
  MenuItem,
  useTheme,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import MenuIcon from '@mui/icons-material/Menu';
import { useAuth } from '../contexts/AuthContext';
import AdminPanelSettingsIcon from '@mui/icons-material/AdminPanelSettings';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';

function Header() {
  const { isAuthenticated, logout, isAdmin } = useAuth();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [anchorEl, setAnchorEl] = useState(null);
  const navigate = useNavigate();

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleClose();
  };

  const handleNavigate = (path) => {
    navigate(path);
    handleClose();
  };

  return (
    <AppBar position="static">
      <Container maxWidth="xl">
        <Toolbar>
          <Typography variant="h6" component={RouterLink} to="/" sx={{ flexGrow: 1, textDecoration: 'none', color: 'inherit' }}>
            RAG-сервис
          </Typography>

          {isMobile ? (
            <>
              <IconButton
                size="large"
                edge="end"
                color="inherit"
                aria-label="menu"
                onClick={handleMenu}
              >
                <MenuIcon />
              </IconButton>
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                keepMounted
                transformOrigin={{
                  vertical: 'top',
                  horizontal: 'right',
                }}
                open={Boolean(anchorEl)}
                onClose={handleClose}
              >
                {isAuthenticated ? (
                  <>
                    <MenuItem component={RouterLink} to="/chat" onClick={handleClose}>
                      Чат
                    </MenuItem>
                    <MenuItem component={RouterLink} to="/documents" onClick={handleClose}>
                      Документы
                    </MenuItem>
                    {isAdmin() && (
                      <MenuItem onClick={() => handleNavigate('/dashboard')}>
                        <ListItemIcon>
                          <AdminPanelSettingsIcon fontSize="small" />
                        </ListItemIcon>
                        <ListItemText>Панель администратора</ListItemText>
                      </MenuItem>
                    )}
                    <MenuItem onClick={() => handleNavigate('/profile')}>
                      <ListItemIcon>
                        <AccountCircleIcon fontSize="small" />
                      </ListItemIcon>
                      <ListItemText>Профиль</ListItemText>
                    </MenuItem>
                    <MenuItem onClick={handleLogout}>Выйти</MenuItem>
                  </>
                ) : (
                  <>
                    <MenuItem component={RouterLink} to="/login" onClick={handleClose}>
                      Войти
                    </MenuItem>
                    <MenuItem component={RouterLink} to="/register" onClick={handleClose}>
                      Регистрация
                    </MenuItem>
                  </>
                )}
              </Menu>
            </>
          ) : (
            <Box sx={{ display: 'flex', gap: 2 }}>
              {isAuthenticated ? (
                <>
                  <Button color="inherit" component={RouterLink} to="/chat">
                    Чат
                  </Button>
                  <Button color="inherit" component={RouterLink} to="/documents">
                    Документы
                  </Button>
                  {isAdmin() && (
                    <Button 
                      color="inherit" 
                      onClick={() => handleNavigate('/dashboard')}
                      startIcon={<AdminPanelSettingsIcon />}
                    >
                      Админ
                    </Button>
                  )}
                  <Button 
                    color="inherit" 
                    onClick={() => handleNavigate('/profile')}
                    startIcon={<AccountCircleIcon />}
                  >
                    Профиль
                  </Button>
                  <Button color="inherit" onClick={logout}>
                    Выйти
                  </Button>
                </>
              ) : (
                <>
                  <Button color="inherit" component={RouterLink} to="/login">
                    Войти
                  </Button>
                  <Button color="inherit" component={RouterLink} to="/register">
                    Регистрация
                  </Button>
                </>
              )}
            </Box>
          )}
        </Toolbar>
      </Container>
    </AppBar>
  );
}

export default Header;