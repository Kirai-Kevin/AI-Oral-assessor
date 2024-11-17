FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN useradd --create-home appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    python3-dev \
    build-essential \
    gcc \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    pkg-config \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/temp_audio && chown -R appuser:appuser /app/temp_audio

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir portaudio==19.7.0 && \
    pip install --no-cache-dir pyaudio==0.2.13 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/taconi/playsound.git

COPY . .

RUN chown -R appuser:appuser /app

USER appuser

# Let Render assign the port
ENV PORT=8501
EXPOSE $PORT

CMD streamlit run --server.port $PORT --server.address 0.0.0.0 app.py