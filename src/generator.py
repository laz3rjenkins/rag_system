import threading

from langchain_community.llms.llamacpp import LlamaCpp
from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

from config.config import (
    LLM_PATH,
    LLM_N_CTX,
    LLM_N_THREADS,
    LLM_N_GPU_LAYERS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_REPEAT_PENALTY,
)

# llama_cpp.Llama не потокобезопасна. Эндпоинты работают в пуле потоков FastAPI,
# поэтому доступ к единственному инстансу модели сериализуем этим локом.
_llm_lock = threading.Lock()


def get_llm(model_path: str = LLM_PATH) -> LlamaCpp:
    return LlamaCpp(
        model_path=model_path,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        repeat_penalty=LLM_REPEAT_PENALTY,
        verbose=True,
    )


def generate(llm: LlamaCpp, prompt: str) -> str:
    """Потокобезопасный вызов LLM: один инстанс, сериализованный доступ."""
    with _llm_lock:
        return llm.invoke(prompt)


# def build_prompt(context: str, question: str) -> str:
#     template = """
# Извлеки только ответ на вопрос.
# Никаких пояснений, цифр, символов или форматирования.
#
# Контекст:
# {context}
#
# Вопрос:
# {question}
#
# Краткий точный ответ:
# """
#     return PromptTemplate.from_template(template).format(
#         context=context,
#         question=question
#     )

def build_prompt(context: str, question: str) -> str:
    global tok
    messages = [
        {"role": "system", "content":
            "Отвечай только на основе контекста, кратко, сохраняя обозначения стандартов."},
        {"role": "user", "content":
            f"Контекст:\n{context}\n\nВопрос:\n{question}\n\nКраткий точный ответ:"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
