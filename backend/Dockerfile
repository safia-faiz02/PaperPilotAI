FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed model at BUILD time so the container
# starts instantly without any download delay at runtime.
# This runs during `docker compose up --build`, not when the server starts.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY alembic.ini .
COPY ./alembic ./alembic
COPY ./app ./app
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000
CMD ["./start.sh"]
