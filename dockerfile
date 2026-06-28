FROM python:3.11-slim

WORKDIR /app  

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY web.py .
COPY results/resnet50.keras .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/ || exit 1

ENTRYPOINT ["streamlit", "run", "web.py", "--server.port=8501", "--server.address=0.0.0.0"]