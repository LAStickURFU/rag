-- Скрипт для добавления дополнительных колонок метаинформации в таблицу documents
-- Выполнять с правами администратора базы данных

-- Добавляем колонки для хранения расширенной метаинформации о чанкинге

-- Проверяем и добавляем chunk_size
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'chunk_size'
    ) THEN
        ALTER TABLE documents ADD COLUMN chunk_size INTEGER;
    END IF;
END $$;

-- Проверяем и добавляем chunk_overlap
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'chunk_overlap'
    ) THEN
        ALTER TABLE documents ADD COLUMN chunk_overlap INTEGER;
    END IF;
END $$;

-- Проверяем и добавляем embedding_model
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'embedding_model'
    ) THEN
        ALTER TABLE documents ADD COLUMN embedding_model VARCHAR;
    END IF;
END $$;

-- Проверяем и добавляем processing_params (JSON)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'documents' AND column_name = 'processing_params'
    ) THEN
        ALTER TABLE documents ADD COLUMN processing_params JSONB;
    END IF;
END $$; 