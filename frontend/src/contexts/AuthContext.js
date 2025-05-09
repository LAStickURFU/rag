import React, { createContext, useState, useEffect, useContext } from 'react';
import { isAuthenticated, getToken, logout, registerUser } from '../services/api';

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
          setUser({ isAuthenticated: true });
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
      setUser({ isAuthenticated: true, username });
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

  const value = {
    user,
    loading,
    error,
    login: handleLogin,
    logout: handleLogout,
    register: handleRegister,
    isAuthenticated: user?.isAuthenticated || false
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