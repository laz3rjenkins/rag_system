# RAG project (Chroma + LangChain + локальная LLM)

Небольшой учебный проект **RAG (Retrieval-Augmented Generation)**: при первом запуске он скачивает/парсит исходный текст из веб-страницы, режет его на чанки, сохраняет эмбеддинги в **ChromaDB**, а затем отвечает на вопросы, подмешивая найденный контекст в промпт локальной LLM через `LlamaCpp`.

## Что умеет

- **Ингест/индексация**: загрузка статьи с Habr, разбиение на чанки, сохранение в `chroma_db/`.
- **Поиск контекста**: top-\(k=3\) чанка из Chroma по эмбеддингам.
- **Генерация ответа**: `RetrievalQA` + локальная модель через `LlamaCpp`.
- **CLI-режим**: интерактивный цикл вопросов/ответов в терминале.

## Структура проекта

- `main.py` — точка входа (инициализация базы при отсутствии и интерактивный чат)
- `src/ingester.py` — загрузка текста из веба и сохранение в Chroma
- `src/retriever.py` — создание ретривера Chroma
- `src/generator.py` — сборка RAG-цепочки (`RetrievalQA`) и промпт
- `src/utils.py` — создание эмбеддингов (`HuggingFaceEmbeddings`)
- `config/config.py` — пути и чтение переменных окружения
- `chroma_db/` — локальная директория с персистентной БД Chroma (создаётся/наполняется при индексации)
- `data/` — директория под данные (в текущей версии может не использоваться)

## Требования

- **Python 3.11+** (рекомендуется 3.13)
- macOS / Linux / Windows (на Windows иногда сложнее собрать зависимости для локальной LLM)
- Доступ в интернет **для первичного парсинга статьи** и (возможно) скачивания моделей эмбеддингов с HuggingFace

## Установка

1) Создайте и активируйте виртуальное окружение.

```bash
python -m venv venv
source venv/bin/activate
```

2) Установите зависимости.

```bash
pip install -r requirements.txt
```

Примечание: проект использует интеграции LangChain/Chroma/HF-эмбеддингов и локальную LLM через llama.cpp. Если при запуске вы увидите `ModuleNotFoundError` (например, для `langchain_huggingface`, `langchain_chroma`, `langchain_classic` или `llama_cpp`), установите недостающие пакеты через `pip install ...` — зависимости могут отличаться в разных окружениях.

## Настройка (ENV)

Скопируйте пример и заполните переменные:

```bash
cp .env.example .env
```

В `.env` используются:

- **`LLM_PATH`** — путь к локальному файлу модели для llama.cpp (обычно `.gguf`)
  - пример: `LLM_PATH=/Users/you/models/llama-7b.Q4_K_M.gguf`
- **`EMBEDDING_MODEL_NAME`** — название модели эмбеддингов для HuggingFace
  - пример: `EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`

Переменные читаются в `config/config.py` через `python-dotenv`.

## Запуск

### 1) Обычный запуск (индексация при первом запуске)

```bash
python main.py
```

Логика такая:

- если директории `chroma_db/` нет — запускается индексация (`parse_data()`),
- затем поднимается ретривер и RAG-цепочка,
- дальше — интерактивные вопросы в консоли.

Для выхода: `q`, `quit` или `exit`.

### 2) Принудительная переиндексация

Если хотите заново собрать базу (например, поменяли эмбеддинги/коллекцию/чанкование), удалите `chroma_db/` и запустите проект снова:

```bash
rm -rf chroma_db
python main.py
```

Либо можно запустить индексацию напрямую:

```bash
python -m src.ingester
```

## Как это работает (high level)

1) `src/ingester.py`
   - грузит страницу `https://habr.com/ru/articles/1011426/` (`WebBaseLoader`)
   - парсит только нужные блоки HTML (через `bs4.SoupStrainer`)
   - бьёт на чанки `chunk_size=1000`, `chunk_overlap=200`
   - считает эмбеддинги и сохраняет документы в Chroma (`collection_name="rag_prompt_context"`)

2) `src/retriever.py`
   - открывает Chroma из `chroma_db/`
   - возвращает `as_retriever(search_kwargs={"k": 3})`

3) `src/generator.py`
   - поднимает `LlamaCpp(model_path=LLM_PATH, n_ctx=4096, n_threads=8, temperature=0.1)`
   - строит `RetrievalQA` с промптом “используй только контекст; если ответа нет — скажи”

4) `main.py`
   - соединяет всё вместе и запускает CLI-цикл.
