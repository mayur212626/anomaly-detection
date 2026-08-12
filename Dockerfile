# Slim serving image for the /score API. Only the runtime deps + the trained
# artifacts — no pyspark/torch/dash/mlflow (pipeline & dashboard only).
FROM python:3.13-slim

WORKDIR /app

# deps first for layer caching
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# app code + the trained artifacts the API loads at startup
COPY api/ ./api/
COPY models/isolation_forest.pkl models/scaler.pkl ./models/
COPY data/features.json ./data/

EXPOSE 8000
# Render provides $PORT; default to 8000 for local `docker run`
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
