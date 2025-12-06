# Семантический поиск с OpenAI Embeddings и Pinecone

Мини-проект, демонстрирующий создание эмбеддингов текстов через Proxy API / OpenAI и семантический поиск по векторному индексу Pinecone.

## Что умеет
- Создать эмбеддинги для набора фраз и загрузить их в индекс Pinecone (`python src/demo_semantic_search.py`).
- Выполнить 3–5 готовых запросов и сохранить результаты в `results/search_runs.md`.
- Делать интерактивные запросы без перезаписи прежних результатов (`python src/interactive_search.py`), новые результаты пишутся в отдельный файл с меткой времени.

## Структура
```
src/
  embeddings.py          # получение эмбеддингов через OpenAI (Proxy API)
  pinecone_client.py     # обёртка для работы с индексом Pinecone
  demo_semantic_search.py# сценарий загрузки фраз и демо-поиска
  interactive_search.py  # интерактивный поиск без перезаписи результатов
data/
  phrases.txt            # корпус фраз (≈200+ про Nissan Juke и VW Beetle)
results/
  search_runs.md         # результаты демо-запросов
  search_run_<ts>.md     # результаты интерактивных запросов
requirements.txt
README.md
```

## Подготовка
1. Python 3.10+ и активированное виртуальное окружение.
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Создайте `.env` (можно начать с шаблона `.env.example`) и заполните ключи:
   ```env
   OPENAI_KEY=...
   OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   PINECONE_API_KEY=...
   PINECONE_ENVIRONMENT=aws:us-east-1
   PINECONE_INDEX_NAME=semantic-cars
   PINECONE_EMBEDDING_DIM=1536
   PINECONE_METRIC=cosine
   ```
   > Индекс с dimension 1536 и cosine-метрикой можно создать вручную в консоли Pinecone или позволить скрипту создать его автоматически.

## Запуск
- Демо-загрузка и поиск:
  ```bash
  python src/demo_semantic_search.py
  ```
  Результаты будут в консоли и `results/search_runs.md`.

- Интерактивный поиск (по уже загруженному индексу):
  ```bash
  python src/interactive_search.py
  ```
  Введите несколько запросов, затем пустую строку. Результаты сохранятся в `results/search_run_<timestamp>.md`.

## Полезные заметки
- `.env` не коммитим; ключи храним только локально.
- Если хотите разделить наборы данных, используйте `namespace` в методах `upsert_texts` и `query` (`src/pinecone_client.py`).
- Данные в `data/phrases.txt` можно заменить на свой корпус (одна фраза — одна строка).

## Лицензия
Проект распространяется под лицензией MIT (см. `LICENSE`).

## Автор
Георгий Белянин (Georgy Belyanin) — georgy.belyanin@gmail.com
