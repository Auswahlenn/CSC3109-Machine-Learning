FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/model/efficientnet_b0_final_best.keras \
    MODEL_MANIFEST_PATH=/app/model/efficientnet_b0_final_manifest.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY deployment/app.py /app/app.py
COPY results/efficientnet_b0_final_best.keras /app/model/efficientnet_b0_final_best.keras
COPY results/efficientnet_b0_final_manifest.json /app/model/efficientnet_b0_final_manifest.json

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
