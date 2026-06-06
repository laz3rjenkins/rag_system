FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=4

WORKDIR /app

# Системные зависимости + инструменты сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Обновляем pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements_gpu.txt .

# Зависимости БЕЗ llama-cpp-python (его нет в requirements_gpu.txt).
RUN python3 -m pip install --no-cache-dir -r requirements_gpu.txt

# llama-cpp-python с CUDA — из готовых wheel'ов, БЕЗ компиляции из исходников.
# cu124 соответствует базовому образу nvidia/cuda:12.4.1. Это убирает
# ~48-минутную сборку и падение на компиляции tools/mtmd (mtmd-cli.cpp).
RUN python3 -m pip install --no-cache-dir --prefer-binary llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

RUN python3 -m pip install --no-cache-dir sentence-transformers

COPY . .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]