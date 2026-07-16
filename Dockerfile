# Stage 1: Build & Package Caching
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system utilities needed for building packages if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only versions of PyTorch and Torchvision first.
# This prevents pip from downloading heavy CUDA/cuDNN GPU binaries (saving ~1.5 GB of footprint).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime Environment
FROM python:3.11-slim

WORKDIR /app

# Install system runtime dependencies for OpenCV and EasyOCR (GLib, OpenGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model files and application source
COPY models/ ./models/
COPY src/ ./src/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ENV=production
ENV ACTIVE_BACKEND=onnx

# Warmup EasyOCR by pre-downloading English detection and recognition models
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Expose port
EXPOSE 8000

# Run API server with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
