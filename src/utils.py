from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config.config import EMBEDDING_MODEL_PATH, EMBED_QUERY_PREFIX, EMBED_PASSAGE_PREFIX


class PrefixedEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings, добавляющий инструкционные префиксы.

    Модели семейства e5 (intfloat/multilingual-e5-*) обучены с префиксами
    "query: " для поисковых запросов и "passage: " для индексируемых текстов.
    Без них релевантность поиска заметно ниже. Префиксы применяются
    автоматически и на индексации (embed_documents), и на поиске (embed_query).
    """

    query_prefix: str = ""
    passage_prefix: str = ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([f"{self.passage_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"{self.query_prefix}{text}")


@lru_cache(maxsize=1)
def get_embeddings() -> PrefixedEmbeddings:
    # lru_cache: модель эмбеддингов тяжёлая, грузим один раз на процесс.
    return PrefixedEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        query_prefix=EMBED_QUERY_PREFIX,
        passage_prefix=EMBED_PASSAGE_PREFIX,
    )
