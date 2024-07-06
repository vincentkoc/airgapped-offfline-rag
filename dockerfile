# Use a multi-stage build for better compatibility
FROM python:3.9-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Check if models directory is empty and fail if it is
RUN if [ -z "$(ls -A models)" ]; then \
    echo "Error: models directory is empty" && exit 1; \
    fi

# Create final image
FROM python:3.9-slim

# Install libgomp1
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy built dependencies and project files from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Set work directory
WORKDIR /app

# Create directories for models and database
RUN mkdir -p models chroma_db

# Copy models and documents
COPY models models
COPY documents documents

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Create a non-root user
# RUN useradd -m myuser
# USER myuser

# Expose Streamlit port (this is just a documentation, it doesn't actually publish the port)
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501
