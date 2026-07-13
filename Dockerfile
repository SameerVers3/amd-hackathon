FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG ANTHROPIC_API_KEY
ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV WHISPER_CACHE_DIR=/app/models
RUN python -c 'from faster_whisper import WhisperModel; print("Pre-downloading whisper tiny model"); WhisperModel("tiny", device="cpu", compute_type="int8", download_root="/app/models")'

COPY setconfig .
COPY src/ ./src/

RUN mkdir -p /input /output
RUN chmod 777 /input /output

CMD ["python", "src/main.py"]
