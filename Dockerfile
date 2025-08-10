FROM python:3.11-slim

# System deps (if we later add audio libs, we can extend here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.trackB.txt requirements.stt.txt /app/
RUN pip install -U pip && pip install -r requirements.trackB.txt -r requirements.stt.txt

COPY . /app

ENV PORT=8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
