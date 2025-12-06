"""Обёртка над Pinecone для загрузки фраз и семантического поиска."""

from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from embeddings import get_embedding

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "semantic-cars")
PINECONE_EMBEDDING_DIM = int(os.getenv("PINECONE_EMBEDDING_DIM", "1536"))
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")


@dataclass
class PineconeConfig:
    api_key: str
    index_name: str
    dimension: int = 1536
    metric: str = "cosine"


class PineconeVectorStore:
    """Утилита для работы с векторным индексом Pinecone."""

    def __init__(self, config: PineconeConfig):
        if not config.api_key:
            raise RuntimeError(
                "Не задан PINECONE_API_KEY. Укажите его в .env или переменных окружения."
            )
        if not config.index_name:
            raise RuntimeError("Имя индекса PINECONE_INDEX_NAME не может быть пустым.")

        self.config = config
        self._pc = Pinecone(api_key=config.api_key)

        self._ensure_index()
        self._index = self._pc.Index(self.config.index_name)
        logger.info("Pinecone индекс '%s' готов к работе", self.config.index_name)

    def _ensure_index(self) -> None:
        """Создать индекс при необходимости."""
        existing_indexes = {getattr(idx, "name", str(idx)) for idx in self._pc.list_indexes()}
        if self.config.index_name in existing_indexes:
            return

        spec = self._build_serverless_spec()
        logger.info(
            "Индекс '%s' не найден. Создаю с размерностью %s и метрикой %s.",
            self.config.index_name,
            self.config.dimension,
            self.config.metric,
        )
        try:
            self._pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=spec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Не удалось создать индекс '%s': %s. Проверьте PINECONE_ENVIRONMENT и регион.",
                self.config.index_name,
                exc,
            )
            raise

    def _build_serverless_spec(self) -> ServerlessSpec:
        """Сформировать ServerlessSpec из переменной окружения или использовать дефолт."""
        env_value = PINECONE_ENVIRONMENT
        if not env_value:
            logger.warning(
                "PINECONE_ENVIRONMENT не указан, использую серверный пресет aws/us-east-1."
            )
            return ServerlessSpec(cloud="aws", region="us-east-1")

        if ":" in env_value:
            cloud, region = env_value.split(":", 1)
        elif "/" in env_value:
            cloud, region = env_value.split("/", 1)
        else:
            cloud = "gcp" if "gcp" in env_value else "aws"
            region = env_value

        return ServerlessSpec(cloud=cloud.strip(), region=region.strip())

    def upsert_texts(self, texts: List[str], namespace: str | None = None) -> List[str]:
        """Создать эмбеддинги для текстов и записать их в индекс."""
        clean_texts = [text.strip() for text in texts if text.strip()]
        if not clean_texts:
            raise ValueError("Список текстов для загрузки пустой.")

        vectors = []
        ids: List[str] = []
        for idx, text in enumerate(clean_texts, start=1):
            vector_id = f"phrase-{idx:03d}"
            embedding = get_embedding(text)
            vectors.append({"id": vector_id, "values": embedding, "metadata": {"text": text}})
            ids.append(vector_id)

        self._index.upsert(vectors=vectors, namespace=namespace)
        logger.info("В индекс '%s' загружено %s фраз.", self.config.index_name, len(ids))
        return ids

    def query(
        self, query_text: str, top_k: int = 5, namespace: str | None = None
    ) -> List[Dict[str, Any]]:
        """Вернуть топ-N ближайших векторов по смыслу."""
        embedding = get_embedding(query_text)
        results = self._index.query(
            vector=embedding, top_k=top_k, include_metadata=True, namespace=namespace
        )

        matches = []
        for match in getattr(results, "matches", []) or []:
            text = ""
            if getattr(match, "metadata", None):
                text = match.metadata.get("text", "")
            matches.append({"id": match.id, "score": match.score, "text": text})

        return matches
