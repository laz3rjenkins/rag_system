import hashlib
import re
from pathlib import Path

import fitz  # pymupdf
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src import utils as utils
from config.config import CHROMA_PATH, CHROMA_COLLECTION, DATA_DIR

TOP_SECTION_PATTERN = r'(?=^\d+\.?\s+[А-ЯA-ZЁ])'

# Колонтитул/футер вида "ГОСТ ISO 9612-2016" — строка, целиком состоящая из
# обозначения ГОСТ (возможно с годом). Обобщено, не привязано к одному документу.
RUNNING_HEADER_PATTERN = re.compile(
    r'^ГОСТ\s+(?:ISO|ИСО|Р|EN)?\s*[\d.\-—:]+(?:\s*[-—]\s*\d{2,4})?$',
    re.IGNORECASE,
)


def split_top_sections(text: str):
    pattern = TOP_SECTION_PATTERN
    sections = re.split(pattern, text, flags=re.MULTILINE)

    # т.к. re.split убирает разделитель, нужно его добавить обратно
    top_titles = re.findall(pattern, text, flags=re.MULTILINE)
    results = []
    for idx, s in enumerate(sections[1:], start=0):
        title = top_titles[idx].strip()
        content = s.strip()
        results.append(f"{title}\n{content}")
    return results


def clean_page_text(text: str) -> str:
    text = text.replace("\xa0", " ")

    lines = []
    for line in text.splitlines():
        line = line.strip()

        # убираем пустые
        if not line:
            continue

        # номер страницы
        if re.fullmatch(r"\d+", line):
            continue

        # колонтитул/футер с обозначением ГОСТ (для любого документа)
        if RUNNING_HEADER_PATTERN.match(line):
            continue

        lines.append(line)

    return "\n".join(lines)


def detect_topic(section_text: str) -> str:
    t = section_text.lower()

    if "средства измер" in t or "оборудован" in t:
        return "equipment"

    if "проведение измер" in t or "испытан" in t:
        return "procedure"

    if "формула" in t or "расчет" in t:
        return "calculation"

    return "general"


def extract_gost_name(text: str) -> str:
    match = re.search(r'ГОСТ\s+(?:ISO|ИСО)?\s*[\d\-—]+', text, re.IGNORECASE)
    if match:
        gost_name = match.group(0)
        gost_name = str.replace(gost_name, '\n', '')

        return gost_name.strip()
    return "UNKNOWN"


def extract_documents_from_pdf(path: str):
    doc = fitz.open(path)

    cleaned_pages = []

    for page in doc:
        raw = page.get_text("text")
        cleaned_pages.append(clean_page_text(raw))

    full_text = "\n".join(cleaned_pages)

    sections = split_top_sections(full_text)

    for i in range(len(sections)):
        sections[i] = sections[i].replace("\n", " ").replace("­ ", "")

    gost_name = extract_gost_name(full_text)
    documents = []

    for idx, section in enumerate(sections):
        lines = [x.strip() for x in section.split("\n") if x.strip()]
        title = " ".join(lines[:2])[:200]

        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": path,
                    "gost": gost_name,
                    "section_index": idx,
                    "section_title": title,
                    "topic": detect_topic(section),
                }
            )
        )

    return documents


def _doc_id(source: str, section_index: int) -> str:
    # Детерминированный id: один и тот же раздел одного файла всегда даёт один id.
    return hashlib.md5(f"{source}|{section_index}".encode("utf-8")).hexdigest()


def index_pdf(vectorstore: Chroma, path: str) -> int:
    documents = extract_documents_from_pdf(path)
    if not documents:
        print(f"[ingest] {path}: разделов не найдено, пропуск")
        return 0

    ids = [_doc_id(path, d.metadata["section_index"]) for d in documents]

    # Дедуп: удаляем прежние вектора этого источника перед повторной индексацией
    # (корректно обрабатывает изменённые и удалённые разделы).
    existing = vectorstore.get(where={"source": path})
    if existing and existing.get("ids"):
        vectorstore.delete(ids=existing["ids"])

    vectorstore.add_documents(documents, ids=ids)
    print(f"[ingest] {path}: проиндексировано разделов: {len(documents)}")
    return len(documents)


def parse_data():
    data_dir = Path(DATA_DIR)
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[ingest] В {data_dir} не найдено PDF-файлов")
        return

    embeddings = utils.get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )

    total = 0
    for pdf in pdf_files:
        total += index_pdf(vectorstore, str(pdf))

    print(f"[ingest] Готово. Файлов: {len(pdf_files)}, разделов всего: {total}")


if __name__ == "__main__":
    parse_data()
