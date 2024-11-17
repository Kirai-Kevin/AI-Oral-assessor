# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN useradd --create-home appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    python3-dev \
    build-essential \
    gcc \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directory for audio files
RUN mkdir -p /app/temp_audio && chown -R appuser:appuser /app/temp_audio

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    git+https://github.com/taconi/playsound.git

# Copy application code
COPY . .

# Change ownership of application files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Initialize environment variables and run app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]