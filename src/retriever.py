
import utils
from langchain_chroma import Chroma
from  langchain_core.vectorstores.base import VectorStoreRetriever
from config.config import CHROMA_PATH


def get_retriever(persist_directory=CHROMA_PATH) -> VectorStoreRetriever:
    embeddings = utils.get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag_prompt_context"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})