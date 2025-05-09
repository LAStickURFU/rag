import requests
import psycopg2
import time

SERVICES = [
    {
        "name": "FastAPI backend",
        "url": "http://localhost:8000/api/healthcheck",
        "type": "http"
    },
    {
        "name": "Frontend (React)",
        "url": "http://localhost:3000",
        "type": "http"
    },
    {
        "name": "Ollama",
        "url": "http://localhost:11434/api/tags",
        "type": "http"
    },
    {
        "name": "Qdrant",
        "url": "http://localhost:6333/collections",
        "type": "http"
    },
]

POSTGRES = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "mysecretpassword",
    "dbname": "postgres"
}

def check_http(name, url):
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code < 400:
            print(f"[OK] {name}: {url} (status {resp.status_code})")
        else:
            print(f"[FAIL] {name}: {url} (status {resp.status_code})")
    except Exception as e:
        print(f"[FAIL] {name}: {url} ({e})")

def check_postgres(cfg):
    try:
        conn = psycopg2.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            dbname=cfg["dbname"],
            connect_timeout=3
        )
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        print(f"[OK] PostgreSQL: {cfg['host']}:{cfg['port']} db={cfg['dbname']}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[FAIL] PostgreSQL: {cfg['host']}:{cfg['port']} db={cfg['dbname']} ({e})")

def main():
    print("Проверка сервисов...")
    for s in SERVICES:
        check_http(s["name"], s["url"])
    check_postgres(POSTGRES)

if __name__ == "__main__":
    main() 