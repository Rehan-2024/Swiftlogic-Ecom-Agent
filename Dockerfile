FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "pydantic>=2.0.0" \
    openai \
    openenv-core \
    "numpy>=1.24"

ENV PYTHONPATH="/app"

RUN ls -la /app/configs /app/env

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
