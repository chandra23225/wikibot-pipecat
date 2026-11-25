FROM python:3.12-slim

# System deps for Whisper (ffmpeg) and Kokoro (espeak-ng for phonemizer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy ALL server files into the container
COPY . .

# Install production deps
RUN uv pip install --system \
    "pipecat-ai[daily,google,kokoro,runner,silero,whisper]>=1.12.0" \
    "wikipedia-api>=0.16.0" \
    "python-dotenv"

# Pre-download Whisper base model so first user doesn't wait
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Render injects PORT env var
EXPOSE 8000

CMD ["sh", "-c", "python bot.py -t daily --host 0.0.0.0 --port ${PORT:-8000}"]
