FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
