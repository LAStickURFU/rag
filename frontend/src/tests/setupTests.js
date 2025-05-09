// Добавляем jest-dom для дополнительных матчеров проверки DOM
import '@testing-library/jest-dom';
import '@testing-library/jest-dom/extend-expect';

// Мок для localStorage
const localStorageMock = (function() {
  let store = {};
  return {
    getItem: jest.fn(key => store[key] || null),
    setItem: jest.fn((key, value) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn(key => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

// Мок для matchMedia (требуется для некоторых компонентов Material UI)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // устаревший
    removeListener: jest.fn(), // устаревший
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Мок для scrollIntoView (требуется для тестов ChatPage)
Element.prototype.scrollIntoView = jest.fn();

// Мок для HTMLElement
if (typeof window.HTMLElement !== 'undefined') {
  Object.defineProperty(window.HTMLElement.prototype, 'scrollIntoView', {
    writable: true,
    value: jest.fn()
  });
}

// Подавляем ошибки консоли во время тестов
const originalConsoleError = console.error;
console.error = (...args) => {
  if (
    /Warning: ReactDOM.render is no longer supported in React 18./.test(args[0]) ||
    /Warning: useLayoutEffect does nothing on the server/.test(args[0]) ||
    /Failed prop type/.test(args[0]) ||
    /Warning: validateDOMNesting/.test(args[0]) ||
    /Warning: `ReactDOMTestUtils.act` is deprecated/.test(args[0])
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Глобальный мок для контекста авторизации
jest.mock('../contexts/AuthContext', () => {
  return {
    AuthProvider: ({ children }) => children,
    useAuth: () => ({
      login: jest.fn().mockResolvedValue({}),
      logout: jest.fn(),
      register: jest.fn().mockResolvedValue({}),
      loading: false,
      error: null,
      user: null,
      isAuthenticated: false
    })
  };
});