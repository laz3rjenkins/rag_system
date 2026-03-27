from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from  langchain_core.vectorstores.base import VectorStoreRetriever


def get_RAG_chain(retriever: VectorStoreRetriever, model_path: str):
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        temperature=0.1,
        verbose=False,
    )

    template = """Используй только предоставленный контекст для ответа. 
    Если ответа нет в тексте, так и скажи.

    КОНТЕКСТ: {context}
    ВОПРОС: {question}
    ОТВЕТ:"""

    prompt = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )