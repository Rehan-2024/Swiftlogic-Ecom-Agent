FROM python:3.11-slim

WORKDIR /app

# Copy everything to /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic>=2.0.0 \
    openai \
    openenv-core

# Ensure ecom_env.py is importable (it's in /app root)
ENV PYTHONPATH="/app"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
