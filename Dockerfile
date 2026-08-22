FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY params.yaml .

# CI bakes the "Production" model (from the MLflow registry) into deploy images.
# Locally, without a model file, the API serves a deterministic dummy model,
# so the full system still works end-to-end.
RUN mkdir -p models data/logs

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
