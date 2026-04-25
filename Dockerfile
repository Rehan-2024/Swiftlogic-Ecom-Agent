FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    DEMO_PATH=/demo \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --skip-repo

COPY requirements.txt /app/requirements.txt
# 1) Lightweight deps from PyPI.
# 2) torch (CPU-only wheel) from PyTorch's CPU index - skips the ~2GB CUDA stack.
# 3) transformers / peft / accelerate / huggingface_hub built on top of the CPU torch.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.1+cpu" \
    && pip install --no-cache-dir \
        "transformers>=4.42,<5" \
        "peft>=0.11,<1" \
        "accelerate>=0.30,<1" \
        "huggingface_hub>=0.23,<1"

COPY . /app

RUN python -c "import json, pathlib; [json.loads(p.read_text(encoding='utf-8')) for p in pathlib.Path('/app/configs').glob('*.json')]" \
    && python -c "from server.app import create_app; create_app()" \
    && python -c "from demo.app import build_demo; build_demo()"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn demo.entry:app --host 0.0.0.0 --port ${PORT}"]
