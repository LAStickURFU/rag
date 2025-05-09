module.exports = {
  // Директории с тестами
  testMatch: ['**/__tests__/**/*.js', '**/?(*.)+(spec|test).js'],
  
  // Игнорируемые директории
  testPathIgnorePatterns: ['/node_modules/', '/build/'],
  
  // Расширения файлов
  moduleFileExtensions: ['js', 'jsx', 'json'],
  
  // Настройка мока для стилей и файлов
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': '<rootDir>/src/tests/__mocks__/styleMock.js',
    '\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$': 
      '<rootDir>/src/tests/__mocks__/fileMock.js',
  },
  
  // Настройка окружения
  testEnvironment: 'jsdom',
  
  // Установка таймаута для тестов
  testTimeout: 10000,
  
  // Охват тестами
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{js,jsx}',
    '!src/tests/**',
    '!src/reportWebVitals.js',
    '!src/index.js',
  ],
  coverageDirectory: 'coverage',
  
  // Настройка Jest для работы с React
  setupFilesAfterEnv: ['<rootDir>/src/tests/setupTests.js'],
  
  // Трансформация файлов
  transform: {
    '^.+\\.(js|jsx)$': 'babel-jest',
  },
  
  // Трансформация для поддержки модулей ES
  transformIgnorePatterns: [
    '/node_modules/(?!(axios)/)'
  ],
}; 