# =============================================================================
# GrammaCore Dockerfile for Hugging Face Spaces (Docker SDK)
# =============================================================================
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

# Install python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python3 -m spacy download en_core_web_sm

# Pre-download Hugging Face model weights at build time
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('vennify/t5-base-grammar-correction'); \
    AutoModelForSeq2SeqLM.from_pretrained('vennify/t5-base-grammar-correction')"

# Copy application files
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Start FastAPI application
CMD ["uvicorn", "grammacore_backend:app", "--host", "0.0.0.0", "--port", "7860"]
