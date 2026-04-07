import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src import utils as utils
from config.config import CHROMA_PATH


def get_text_from_web():
    bs4_strainer = bs4.SoupStrainer(class_=("tm-title tm-title_h1", "article-body"))

    loader = WebBaseLoader(
        web_path=("https://habr.com/ru/articles/1011426/"),
        bs_kwargs={"parse_only": bs4_strainer},
    )

    return loader.load()


def parse_data():
    return
    # todo: сделать проверку на существовании статьи в базе, чтобы данные не дублирвоались
    docs = get_text_from_web()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(docs)

    embeddings = utils.get_embeddings()

    vector_store = Chroma(
        collection_name="rag_prompt_context",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    ids = vector_store.add_documents(all_splits)


if __name__ == '__main__':
    parse_data()