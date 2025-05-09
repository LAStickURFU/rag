/**
 * Скрипт для запуска всех тестов в пакетном режиме
 * Запуск: node src/runAllTests.js
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Директория с тестами
const testsDir = path.join(__dirname, 'tests');

// Найти все тестовые файлы рекурсивно
function findTestFiles(dir) {
  let results = [];
  const items = fs.readdirSync(dir);
 
  for (const item of items) {
    const itemPath = path.join(dir, item);
    const stat = fs.statSync(itemPath);
   
    if (stat.isDirectory()) {
      results = results.concat(findTestFiles(itemPath));
    } else if (item.endsWith('.test.js')) {
      results.push(itemPath);
    }
  }
 
  return results;
}

// Запуск тестов
try {
  console.log('🔍 Поиск тестовых файлов...');
  const testFiles = findTestFiles(testsDir);
  console.log(`📋 Найдено ${testFiles.length} тестовых файлов`);
 
  console.log('\n🧪 Запуск всех тестов в пакетном режиме...\n');
 
  // Запуск тестов в пакетном режиме
  execSync(`npx jest --config=jest.config.js`, { stdio: 'inherit' });
 
  console.log('\n✅ Все тесты выполнены!');
} catch (error) {
  console.error('\n❌ Ошибка при выполнении тестов:', error.message);
  process.exit(1);
}