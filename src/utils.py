from langchain_huggingface import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL_PATH


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
