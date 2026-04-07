import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

LLM_PATH = os.getenv("LLM_PATH", "")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME", "")

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_USERNAME = os.getenv("DB_USERNAME", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")
