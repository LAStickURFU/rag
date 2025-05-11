-- Скрипт для создания администратора по умолчанию
-- Этот скрипт автоматически выполняется при первом запуске контейнера

-- Функция для создания администратора, если нет ни одного пользователя
CREATE OR REPLACE FUNCTION create_default_admin()
RETURNS void AS $$
DECLARE
    user_count integer;
BEGIN
    -- Проверяем наличие таблицы users
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
        -- Подсчитываем количество пользователей
        SELECT COUNT(*) INTO user_count FROM users;
        
        -- Если пользователей нет, создаем администратора
        IF user_count = 0 THEN
            -- Пароль: 'admin123' (хеш с солью для bcrypt)
            INSERT INTO users (username, hashed_password, disabled, role, created_at, email, is_active)
            VALUES (
                'admin',
                '$2b$12$N0j/LBU3Dd5UlKiDAw4JYuMOlNiLiPjA9EZ4P7/vDc5S6e.5UXrLG',
                false,
                'admin',
                NOW(),
                'admin@example.com',
                true
            );
            
            RAISE NOTICE 'Администратор по умолчанию создан: admin / admin123';
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Вызываем функцию для создания администратора
SELECT create_default_admin();

-- Очищаем функцию после использования
DROP FUNCTION create_default_admin(); 