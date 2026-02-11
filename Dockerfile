# UncertaintyML Platform
# Multi-stage Dockerfile: api | dashboard

# ─── Base ───
FROM python:3.10-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models
ENV PYTHONUNBUFFERED=1

# ─── API target ───
FROM base AS api
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Dashboard target ───
FROM base AS dashboard
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
