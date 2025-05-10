-- Скрипт для добавления колонки chunking_mode в таблицу documents
-- Выполнять с правами администратора базы данных

-- Проверяем, существует ли колонка, чтобы избежать ошибок
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'chunking_mode'
    ) THEN
        -- Добавляем колонку с значением по умолчанию 'character'
        ALTER TABLE documents ADD COLUMN chunking_mode VARCHAR DEFAULT 'character';
        
        -- Заполняем существующие записи
        UPDATE documents SET chunking_mode = 'character' WHERE chunking_mode IS NULL;
    END IF;
END $$; 