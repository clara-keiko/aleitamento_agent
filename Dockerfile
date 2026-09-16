FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY main.py ingest_openai_kb.py ./

# Não roda como root: se o container for comprometido, limita o estrago.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 10000

# $PORT vem da plataforma (Render, Fly, Cloud Run); 10000 é só o fallback.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
