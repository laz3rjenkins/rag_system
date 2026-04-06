from src import utils as utils
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from config.config import CHROMA_PATH

def get_retriever() -> VectorStoreRetriever:
    embeddings = utils.get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="rag_prompt_context"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 8})

def detect_query_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["оборуд", "прибор", "средства измер"]):
        return "equipment"

    if any(x in q for x in ["как провести", "испытан", "методика", "порядок"]):
        return "procedure"

    if any(x in q for x in ["расчет", "формула", "погрешност", "неопределен"]):
        return "calculation"

    return "general"


def smart_retrieve(vectorstore: Chroma, query: str):
    intent = detect_query_intent(query)

    k = 2 if intent != "general" else 2

    if intent in ["equipment", "procedure", "calculation"]:
        docs = vectorstore.similarity_search(
            query,
            k=k,
            filter={"topic": intent}
        )

        # fallback если filter слишком жесткий
        if not docs:
            docs = vectorstore.similarity_search(query, k=k)
    else:
        docs = vectorstore.similarity_search(query, k=k)

    return docs