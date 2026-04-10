from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

from config.config import DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

MARIADB_URL = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(
    MARIADB_URL,
    pool_recycle=3600,
    pool_pre_ping=True,
    pool_timeout=30,
    max_overflow=10
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Chat(Base):
    __tablename__ = "chats"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(String(36), ForeignKey("chats.id"))
    sender = Column(String(20))  # 'user' или 'bot'
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")


# Функция инициализации (создаст таблицы, если их нет)
def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
