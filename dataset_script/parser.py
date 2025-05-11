import os
import re
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from playwright.async_api import async_playwright

# Константы путей
HTML_DIR = Path("html")
TEXT_DIR = Path("text")
SUCCESS_LOG = "success.log"
ERROR_LOG = "error.log"
RETRIES_LOG = "retries.log"

# Константы ожиданий
WAIT_FOR_SELECTOR_TIMEOUT = 3000  # ms
WAIT_FOR_FALLBACK_SELECTOR_TIMEOUT = 3000  # ms

# Количество одновременных задач
MAX_CONCURRENT_TASKS = 10

# Создание директорий
HTML_DIR.mkdir(exist_ok=True)
TEXT_DIR.mkdir(exist_ok=True)

# Настройка логов
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
success_urls = set()

if os.path.exists(SUCCESS_LOG):
    with open(SUCCESS_LOG, encoding="utf-8") as f:
        success_urls = set(f.read().splitlines())


def clean_filename(url: str) -> str:
    endpoint = url.rstrip("/").split("/")[-1] or "index"
    return re.sub(r'\W+', '-', endpoint).strip("-")


def clean_text(text):
    text = text.replace('\xa0', ' ')
    return re.sub(r'[^\x20-\x7E\u0400-\u04FF\u2010-\u202F\u00A0-\u00FF]', '', text).strip()


def parse_text_from_html(html_path: Path) -> str:
    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    article_body = soup.find("div", itemprop="articleBody")
    if article_body:
        content_blocks = article_body.find_all(["p", "h2", "h3", "ul", "ol"])
    else:
        logging.warning("[WARN] articleBody не найден, пробуем div.baseEditorStyles")
        fallback = soup.find("div", class_="baseEditorStyles")
        content_blocks = fallback.find_all(["p", "h2", "h3", "ul", "ol"]) if fallback else []

    if not content_blocks:
        logging.warning("[WARN] Не найдено ни одного содержимого в статье")

    output_lines = []
    for tag in content_blocks:
        if tag.name in ["p", "h2", "h3"]:
            text = clean_text(tag.get_text())
            if text:
                output_lines.append(text)
        elif tag.name in ["ul", "ol"]:
            items = tag.find_all("li")
            for li in items:
                line = clean_text(li.get_text())
                if line:
                    output_lines.append(f"• {line}")

    article_text = "\n".join(output_lines)
    return article_text


def parse_sitemap(sitemap_path: str):
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    return [url.find('sm:loc', ns).text for url in root.findall('sm:url', ns)]


async def fetch_and_process_url(url: str):
    if url in success_urls:
        logging.info(f"[SKIP] Уже обработан: {url}")
        return

    max_retries = 3
    delay_between_retries = 5

    for attempt in range(1, max_retries + 1):
        logging.info(f"[TRY {attempt}] Обработка: {url}")
        with open(RETRIES_LOG, "a", encoding="utf-8") as retry_log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            retry_log.write(f"[{timestamp}] Попытка {attempt} — {url}\n")

        try:
            filename = clean_filename(url)
            html_file = HTML_DIR / f"{filename}.html"
            txt_file = TEXT_DIR / f"{filename}.txt"

            logging.info(f"[START] Загрузка: {url}")
            total_start = time.time()

            async with async_playwright() as p:
                start = time.time()
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-infobars",
                        "--no-sandbox",
                        "--disable-dev-shm-usage"
                    ]
                )
                logging.info(f"[TIME] Инициализация браузера - {time.time() - start:.2f} сек")

                start = time.time()
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                    bypass_csp=True,
                    java_script_enabled=True
                )
                page = await context.new_page()
                await page.goto(url, timeout=60000)
                logging.info(f"[TIME] Загрузка страницы - {time.time() - start:.2f} сек")

                start = time.time()
                try:
                    await page.wait_for_selector("div[itemprop='articleBody']", timeout=WAIT_FOR_SELECTOR_TIMEOUT)
                except Exception:
                    logging.warning("[FALLBACK] articleBody не найден, ожидаем baseEditorStyles")
                    await page.wait_for_selector("div.baseEditorStyles", timeout=WAIT_FOR_FALLBACK_SELECTOR_TIMEOUT)
                logging.info(f"[TIME] Ожидание появления контента - {time.time() - start:.2f} сек")

                start = time.time()
                content = await page.content()
                logging.info(f"[TIME] Получение HTML - {time.time() - start:.2f} сек")

                await browser.close()

            logging.info(f"[DONE] Страница загружена: {url}")

            start = time.time()
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(content)
            logging.info(f"[TIME] Сохранение HTML - {time.time() - start:.2f} сек")

            logging.info(f"[PARSING] Парсинг: {url}")
            start = time.time()
            parsed_text = parse_text_from_html(html_file)
            if not parsed_text.strip() or len(parsed_text) < 200:
                with open("suspicious.log", "a", encoding="utf-8") as log:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log.write(f"[{timestamp}] Короткий или пустой текст: {url} (длина {len(parsed_text)} символов)")
                suspicious_dir = Path("suspicious_html")
                suspicious_dir.mkdir(exist_ok=True)
                suspicious_file = suspicious_dir / f"{clean_filename(url)}_{timestamp.replace(':', '-')}.html"
                with open(suspicious_file, "w", encoding="utf-8") as sf:
                    sf.write(content)
                logging.info(f"[DUMP] Подозрительная страница сохранена: {suspicious_file}")
                logging.warning(
                    f"[WARN] Маленький или пустой текст на странице: {url} (длина {len(parsed_text)} символов)")
            logging.info(f"[TIME] Парсинг текста - {time.time() - start:.2f} сек")

            start = time.time()
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(parsed_text)
            logging.info(f"[TIME] Сохранение текста - {time.time() - start:.2f} сек")

            with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{timestamp}] Успешно обработан за {attempt} попытку(и): {url}\n")

            logging.info(f"[LOG] Успешно обработан: {url} за {attempt} попытку(и)")
            logging.info(f"[TIME] Общее время обработки - {time.time() - total_start:.2f} сек")
            return

        except Exception as e:
            logging.error(f"[ERROR] Попытка {attempt} не удалась: {e}")
            if attempt < max_retries:
                logging.info(f"[WAIT] Повтор через {delay_between_retries} секунд...")
                await asyncio.sleep(delay_between_retries)
            else:
                with open(ERROR_LOG, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] Не удалось после {max_retries} попыток: {url}\n")
                logging.error(f"[FAILED] Не удалось обработать после {max_retries} попыток: {url}")
                failed_dir = Path("failed_html")
                failed_dir.mkdir(exist_ok=True)
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        context = await browser.new_context()
                        page = await context.new_page()
                        await page.goto(url, timeout=60000)
                        failed_html = await page.content()
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        failed_file = failed_dir / f"{clean_filename(url)}_{timestamp}.html"
                        with open(failed_file, "w", encoding="utf-8") as ff:
                            ff.write(failed_html)
                        await browser.close()
                        logging.info(f"[DUMP] Содержимое неудачной страницы сохранено: {failed_file}")
                except Exception as save_err:
                    logging.warning(f"[DUMP] Не удалось сохранить HTML неудачной страницы: {save_err}")


async def main(sitemap_path, resume_from=None):
    urls = parse_sitemap(sitemap_path)
    if resume_from and resume_from in urls:
        urls = urls[urls.index(resume_from):]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def sem_task(url):
        async with semaphore:
            await fetch_and_process_url(url)

    tasks = [sem_task(url) for url in urls]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sitemap", required=True, help="Путь к sitemap.xml")
    parser.add_argument("--resume-from", help="URL, с которого продолжить")
    args = parser.parse_args()

    asyncio.run(main(args.sitemap, args.resume_from))
