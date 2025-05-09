import psycopg2
import time

PG_HOST = "localhost"
PG_PORT = 5432
PG_SUPERUSER = "postgres"
PG_SUPERPASS = ""  # если нужен пароль для postgres, укажи здесь
NEW_USER = "raguser"
NEW_PASS = "ragpass"
NEW_DB = "ragdb"

def wait_for_postgres():
    for _ in range(30):
        try:
            conn = psycopg2.connect(
                dbname="postgres",
                user=PG_SUPERUSER,
                password=PG_SUPERPASS,
                host=PG_HOST,
                port=PG_PORT,
            )
            conn.close()
            return True
        except Exception as e:
            print("Postgres не готов, жду...")
            time.sleep(2)
    return False

def main():
    if not wait_for_postgres():
        print("Postgres не стартовал, выход.")
        return

    conn = psycopg2.connect(
        dbname="postgres",
        user=PG_SUPERUSER,
        password=PG_SUPERPASS,
        host=PG_HOST,
        port=PG_PORT,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Создать пользователя, если не существует
    cur.execute(f"SELECT 1 FROM pg_roles WHERE rolname='{NEW_USER}'")
    if not cur.fetchone():
        cur.execute(f"CREATE ROLE {NEW_USER} WITH LOGIN PASSWORD '{NEW_PASS}';")
        print(f"Пользователь {NEW_USER} создан.")
    else:
        print(f"Пользователь {NEW_USER} уже существует.")

    # Создать базу, если не существует
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{NEW_DB}'")
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {NEW_DB} OWNER {NEW_USER};")
        print(f"База {NEW_DB} создана.")
    else:
        print(f"База {NEW_DB} уже существует.")

    # Выдать права
    cur.execute(f"GRANT ALL PRIVILEGES ON DATABASE {NEW_DB} TO {NEW_USER};")
    print(f"Права на базу {NEW_DB} выданы пользователю {NEW_USER}.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main() 