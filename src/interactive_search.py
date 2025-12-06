"""Интерактивный поиск без перезаписи результатов.

Запускай из корня: python src/interactive_search.py
Скрипт:
- при старте загружает фразы, делает upsert (перезапись по тем же ID не критична);
- спрашивает запросы в консоли (пустая строка — выход);
- сохраняет результаты в новый Markdown-файл с меткой времени.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

from pinecone_client import PineconeConfig, PineconeVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_phrases(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Файл с фразами не найден: {path}")
    with path.open(encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def build_store() -> PineconeVectorStore:
    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "semantic-cars")
    dimension = int(os.getenv("PINECONE_EMBEDDING_DIM", "1536"))
    metric = os.getenv("PINECONE_METRIC", "cosine")
    config = PineconeConfig(
        api_key=api_key,
        index_name=index_name,
        dimension=dimension,
        metric=metric,
    )
    return PineconeVectorStore(config)


def prompt_queries() -> List[str]:
    print("Введите запросы для семантического поиска (пустая строка — завершить):")
    queries: List[str] = []
    while True:
        query = input("> ").strip()
        if not query:
            break
        queries.append(query)
    return queries


def save_results_md(path: Path, runs: List[Tuple[str, List[dict[str, object]]]]) -> None:
    lines: List[str] = ["# Результаты семантического поиска", ""]
    for idx, (query_text, matches) in enumerate(runs, start=1):
        lines.append(f"## Запрос {idx}: {query_text}")
        lines.append("| # | ID | Score | Текст |")
        lines.append("|---|-----------|-------|-------|")
        for order, match in enumerate(matches, start=1):
            vector_id = match.get("id", "")
            score = match.get("score", 0.0)
            text = match.get("text", "")
            lines.append(f"| {order} | {vector_id} | {score:.4f} | {text} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Результаты сохранены в %s", path)


def main() -> None:
    load_dotenv()

    try:
        store = build_store()
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось инициализировать Pinecone: %s", exc)
        sys.exit(1)

    # Предполагаем, что индекс уже заполнен (без повторного upsert).
    # Если индекс пустой, подскажем, что нужно сначала прогнать demo скрипт.
    try:
        stats = store._index.describe_index_stats()  # type: ignore[attr-defined]
        vector_count = stats.get("total_vector_count", 0)
        if not vector_count:
            logger.error(
                "Индекс пустой. Сначала запустите python src/demo_semantic_search.py для загрузки фраз."
            )
            sys.exit(1)
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось получить статистику индекса, продолжаем поиск вслепую.")

    queries = prompt_queries()
    if not queries:
        print("Запросы не заданы, выходим.")
        return

    runs: List[Tuple[str, List[dict[str, object]]]] = []
    for query_text in queries:
        try:
            matches = store.query(query_text, top_k=5)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка при поиске по запросу '%s': %s", query_text, exc)
            continue

        print(f"\n=== Запрос: {query_text}")
        for idx, match in enumerate(matches, start=1):
            vector_id = match.get("id", "")
            score = match.get("score", 0.0)
            text = match.get("text", "")
            print(f"{idx}. [id={vector_id}, score={score:.4f}] {text}")
        runs.append((query_text, matches))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).resolve().parent.parent / "results" / f"search_run_{timestamp}.md"
    save_results_md(results_path, runs)


if __name__ == "__main__":
    main()
