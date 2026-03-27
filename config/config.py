import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

LLM_PATH = os.getenv("LLM_PATH", "")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_NAME", "")