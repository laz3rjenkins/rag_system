import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import Message, Chat, get_db, init_db
from src.retriever import get_retriever, smart_retrieve
from src.ingester import parse_data
from config.config import LLM_PATH, CHROMA_PATH
from src.generator import get_llm, build_prompt
from langchain_chroma import Chroma


def write_log(context: str, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as file:
        file.write(context)


retriever = None
vectorstore: Chroma | None = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever, vectorstore

    init_db()

    if not os.path.exists(CHROMA_PATH):
        parse_data()

    llm = get_llm(LLM_PATH)
    retriever = get_retriever()
    vectorstore = retriever.vectorstore

    yield


app = FastAPI(lifespan=lifespan, debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    prompt: str
    chat_id: Optional[str] = None


@app.post("/ask")
async def ask_llm(question: Question, db: Session = Depends(get_db)):
    global vectorstore, llm

    target_chat_id = None
    new_chat_title = None

    # 1. ЛОГИКА ЧАТА
    if not question.chat_id:
        # Создаем новый чат, если ID не передан
        new_chat = Chat(title=question.prompt[:50])
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)
        target_chat_id = new_chat.id
        new_chat_title = new_chat.title
    else:
        # Проверяем существование чата
        chat = db.query(Chat).filter(Chat.id == question.chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        target_chat_id = chat.id

    # 2. СОХРАНЯЕМ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ
    user_msg = Message(chat_id=target_chat_id, sender="user", text=question.prompt)
    db.add(user_msg)
    db.commit()

    # 3. ГЕНЕРАЦИЯ ОТВЕТА
    try:
        print(f"Thinking about: {question.prompt}")

        docs = smart_retrieve(vectorstore, question.prompt)
        # print("\n=== RETRIEVED DOCS ===")
        #
        # for i, doc in enumerate(docs):
        #     print(f"--- DOC {i} ---")
        #     print(doc.page_content)
        #     print(doc.metadata)

        context = "\n\n".join(
            doc.page_content for doc in docs  # [:4]
        )

        write_log(context, "context_log.txt")
        print("HERE STARTING TO GET CONTEXT==========================================")
        print(f"Context: {context}")
        print("HERE ENDING TO GET CONTEXT==========================================")

        prompt = build_prompt(context, question.prompt).strip()

        write_log(prompt, "prompt_log.txt")

        bot_text = llm.invoke(prompt)

        # 4. СОХРАНЯЕМ ОТВЕТ БОТА
        bot_msg = Message(chat_id=target_chat_id, sender="bot", text=bot_text)
        db.add(bot_msg)
        db.commit()

        return {
            "answer": bot_text,
            "chat_id": target_chat_id,
            "title": new_chat_title
        }
    except Exception as e:
        db.rollback()
        print(f"Error: {e.with_traceback(None)}")
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для получения списка чатов в сайдбар
@app.get("/chats")
async def get_chats(db: Session = Depends(get_db)):
    chats = db.query(Chat).order_by(Chat.created_at.desc()).all()

    return chats


# Эндпоинt для загрузки истории конкретного чата
@app.get("/chats/{chat_id}")
async def get_chat_history(chat_id: str, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()

    return messages
