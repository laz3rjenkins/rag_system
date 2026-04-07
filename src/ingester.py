import fitz  # pymupdf
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
import utils
from config.config import CHROMA_PATH

TOP_SECTION_PATTERN = r'(?=^\d+\s+[А-ЯA-ZЁ])'


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

        # page number
        if re.fullmatch(r"\d+", line):
            continue

        # header/footer ГОСТ
        if re.search(r"ГОСТ\s+ISO\s+9612", line):
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


def parse_data():
    documents = extract_documents_from_pdf("data/4293750815.pdf")

    # for debug
    # for i, doc in enumerate(documents):
    #     print(f"\n--- CHUNK {i} ---")
    #     print(doc.page_content)
    #
    # exit(0)

    embeddings = utils.get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="rag_prompt_context"
    )

    vectorstore.add_documents(documents)

    print(f"Indexed {len(documents)} sections")


if __name__ == "__main__":
    parse_data()

