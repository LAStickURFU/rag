import React, { createContext, useState, useEffect, useContext } from 'react';
import { isAuthenticated, getToken, logout, registerUser, getCurrentUser } from '../services/api';

// Создаем контекст для авторизации
const AuthContext = createContext();

// Провайдер контекста
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Проверка авторизации при загрузке
  useEffect(() => {
    const checkAuth = async () => {
      try {
        if (isAuthenticated()) {
          // Получаем информацию о пользователе с сервера
          try {
            const userData = await getCurrentUser();
            setUser({
              isAuthenticated: true,
              id: userData.id,
              username: userData.username,
              role: userData.role || 'user', // По умолчанию роль пользователя
              created_at: userData.created_at
            });
          } catch (userError) {
            console.error("Ошибка получения данных пользователя:", userError);
            // Если не удалось получить данные, устанавливаем базовую информацию
            setUser({ isAuthenticated: true });
          }
        }
      } catch (err) {
        console.error("Ошибка проверки авторизации:", err);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  // Авторизация пользователя
  const handleLogin = async (username, password) => {
    setLoading(true);
    setError(null);
    try {
      const data = await getToken(username, password);
      // После успешной авторизации получаем данные пользователя
      try {
        const userData = await getCurrentUser();
        setUser({
          isAuthenticated: true,
          id: userData.id,
          username: userData.username,
          role: userData.role || 'user',
          created_at: userData.created_at
        });
      } catch (userError) {
        console.error("Ошибка получения данных пользователя:", userError);
        setUser({ isAuthenticated: true, username });
      }
      return data;
    } catch (err) {
      setError(err.message || 'Ошибка авторизации');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Выход пользователя
  const handleLogout = () => {
    logout();
    setUser(null);
    setError(null);
  };

  // Регистрация пользователя
  const handleRegister = async (username, password) => {
    setLoading(true);
    setError(null);
    try {
      const data = await registerUser(username, password);
      return data;
    } catch (err) {
      setError(err.message || 'Ошибка регистрации');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Проверка, является ли пользователь администратором
  const isAdmin = () => {
    return user?.role === 'admin';
  };

  const value = {
    user,
    loading,
    error,
    login: handleLogin,
    logout: handleLogout,
    register: handleRegister,
    isAuthenticated: user?.isAuthenticated || false,
    isAdmin: isAdmin,
    userRole: user?.role || 'user'
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Хук для использования контекста
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth должен использоваться внутри AuthProvider');
  }
  return context;
};

export default AuthContext;