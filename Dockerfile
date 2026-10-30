FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ui.py ./
COPY docs/ ./docs/
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/

EXPOSE 8000

CMD ["uvicorn", "finpulse.api:app", "--host", "0.0.0.0", "--port", "8000"]
