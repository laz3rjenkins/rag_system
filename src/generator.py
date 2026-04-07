from config.config import LLM_PATH
from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


def get_llm(model_path: str):
    return LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        temperature=0.0,
        max_tokens=300,
        repeat_penalty=1.0,
        # top_p=0.95,
        # top_k=40,
        n_gpu_layers=-1,
        verbose=True
    )


def build_prompt(context: str, question: str) -> str:
    template = """
Извлеки только ответ на вопрос. 
Никаких пояснений, цифр, символов или форматирования.

Контекст:
{context}

Вопрос:
{question}

Краткий точный ответ:
"""
    return PromptTemplate.from_template(template).format(
        context=context,
        question=question
    )


def ask_llm_without_context(prompt: str):
    model_path = LLM_PATH
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Увеличиваем контекст (Gemini умеет в огромные окна, 4096 — маловато)
        n_threads=8,
        temperature=0.7,  # Gemini обычно работает в диапазоне 0.7-0.8
        max_tokens=2048,  # КЛЮЧЕВОЙ ПАРАМЕТР: разрешаем модели писать длинные ответы
        repeat_penalty=1.18,  # Чтобы не зацикливалась (стандарт для современных LLM)
        top_p=0.95,  # Добавляет "разумности" и разнообразия, как у Google
        top_k=40,
        verbose=False,
    )

    result = llm(
        prompt,
        max_tokens=2048,
        stop=["<END>"]
    )

    output = result["choices"][0]["text"].strip()

    res = {'result': output}
    return res
