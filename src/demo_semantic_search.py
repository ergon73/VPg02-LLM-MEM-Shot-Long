"""Демо сценарий: загрузка фраз в Pinecone и семантический поиск."""

from pathlib import Path
from typing import List, Tuple
import logging
import os
import sys

from dotenv import load_dotenv

from pinecone_client import PineconeConfig, PineconeVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_phrases(path: Path) -> List[str]:
    """Прочитать фразы из файла, отфильтровав пустые строки."""
    if not path.exists():
        raise FileNotFoundError(f"Файл с фразами не найден: {path}")

    with path.open(encoding="utf-8") as file:
        phrases = [line.strip() for line in file.readlines() if line.strip()]
    return phrases


def print_matches(query_text: str, matches: List[dict[str, object]]) -> None:
    """Вывести результаты поиска в консоль."""
    print(f"=== Запрос: {query_text}")
    for idx, match in enumerate(matches, start=1):
        text = match.get("text", "")
        score = match.get("score", 0.0)
        vector_id = match.get("id", "")
        print(f"{idx}. [id={vector_id}, score={score:.4f}] {text}")
    print()


def save_results_md(path: Path, runs: List[Tuple[str, List[dict[str, object]]]]) -> None:
    """Сохранить результаты поиска в Markdown."""
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


def build_store() -> PineconeVectorStore:
    """Инициализировать PineconeVectorStore из переменных окружения."""
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


def main() -> None:
    load_dotenv()

    try:
        store = build_store()
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось инициализировать Pinecone: %s", exc)
        sys.exit(1)

    phrases_path = Path(__file__).resolve().parent.parent / "data" / "phrases.txt"
    try:
        phrases = load_phrases(phrases_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if len(phrases) < 20:
        logger.warning("Фраз всего %s, рекомендуется >= 20.", len(phrases))

    try:
        store.upsert_texts(phrases)
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось загрузить фразы в Pinecone: %s", exc)
        sys.exit(1)

    queries = [
        "Какой автомобиль стал символом эпохи хиппи?",
        "Какой компактный кроссовер подходит для города?",
        "Какая модель часто тюнингуется под ретро стиль?",
        "Какой автомобиль известен как 'народный' во многих странах?",
    ]

    runs: List[Tuple[str, List[dict[str, object]]]] = []
    for query_text in queries:
        try:
            matches = store.query(query_text, top_k=5)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка при поиске по запросу '%s': %s", query_text, exc)
            continue

        print_matches(query_text, matches)
        runs.append((query_text, matches))

    results_path = Path(__file__).resolve().parent.parent / "results" / "search_runs.md"
    save_results_md(results_path, runs)


if __name__ == "__main__":
    main()
