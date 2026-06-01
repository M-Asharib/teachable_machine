FROM python:3.11-slim

# HuggingFace Spaces: create non-root user (required)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install system dependencies (as root first, then switch back)
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    supervisor \
    && rm -rf /var/lib/apt/lists/*
USER user

# Install PyTorch CPU-only (smaller, no CUDA overhead)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    python-multipart==0.0.9 \
    requests==2.32.3 \
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    pillow==10.3.0 \
    streamlit==1.35.0

# Copy application code (chown to user)
COPY --chown=user . /app

# Create necessary runtime directories
RUN mkdir -p /app/backend/dataset /app/backend/logs

# Environment
ENV PYTHONPATH=/app
ENV BACKEND_URL=http://localhost:8000

# Supervisord config
COPY --chown=user supervisord.conf /app/supervisord.conf

# HuggingFace Spaces MUST expose port 7860
EXPOSE 7860

# Run both FastAPI (8000) and Streamlit (7860) via supervisord
CMD ["/usr/bin/supervisord", "-c", "/app/supervisord.conf"]
