"""Утилиты для работы с Embedding API через Proxy API / OpenAI."""

from typing import List
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_openai_client() -> OpenAI:
    """Вернуть сконфигурированный клиент OpenAI (Proxy API)."""
    if not OPENAI_KEY:
        raise RuntimeError(
            "Не задан API-ключ OPENAI_KEY. Укажите его в .env или переменных окружения."
        )
    return OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL)


def get_embedding(text: str, model: str | None = None) -> List[float]:
    """Вернуть эмбеддинг для заданного текста."""
    if not text.strip():
        raise ValueError("Текст для эмбеддинга не должен быть пустым.")

    client = get_openai_client()
    model_name = model or OPENAI_EMBEDDING_MODEL

    try:
        response = client.embeddings.create(model=model_name, input=text)
        return list(map(float, response.data[0].embedding))
    except Exception:
        logger.exception("Не удалось получить эмбеддинг для текста: %s", text)
        raise
