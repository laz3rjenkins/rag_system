import os
from src.retriever import get_retriever
from src.ingester import parse_data
from config.config import LLM_PATH
from src.generator import get_RAG_chain
from config.config import CHROMA_PATH

def main():
    if not os.path.exists(CHROMA_PATH):
        print("База данных не найдена. Начинаю парсинг...")
        parse_data()

    retriever = get_retriever()
    rag_chain = get_RAG_chain(retriever, LLM_PATH)

    print("\n--- RAG is ready! q, exit or quit for exit ---")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break

        print("thinking...")
        response = rag_chain.invoke(query)
        print(f"\nAnswer: {response['result']}")


if __name__ == "__main__":
    main()