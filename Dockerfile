FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including build tools for insightface)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    g++ \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p models/weights logs

# Expose port
EXPOSE 8000

# Run full API application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
