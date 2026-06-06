import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))

LLM_PATH = os.getenv("LLM_PATH", "")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME", "")

# --- ChromaDB ---
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_prompt_context")

# --- Эмбеддинги ---
# Модели семейства e5 (intfloat/multilingual-e5-*) требуют префиксов:
# "query: " для запросов и "passage: " для документов. Без них качество поиска
# заметно падает. Для других моделей префиксы можно отключить пустой строкой.
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "query: ")
EMBED_PASSAGE_PREFIX = os.getenv("EMBED_PASSAGE_PREFIX", "passage: ")

# --- Поиск (retrieval) ---
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# --- LLM (llama.cpp) ---
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "8192"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "8"))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))  # -1 = все слои на GPU, 0 = только CPU
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "300"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.0"))

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_USERNAME = os.getenv("DB_USERNAME", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
