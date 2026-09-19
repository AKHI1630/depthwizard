FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgdal-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY eval/ eval/
COPY scripts/ scripts/

# Model weights download at startup (not baked in)
COPY download_models.py .

ENV HF_HUB_DISABLE_XET=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD python download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 7860
