FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /service

# dependencies first -> this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# inference code + the trained model (raw data and training code are excluded)
COPY app/ ./app/
COPY ml/registry/model.pkl ./ml/registry/model.pkl

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
