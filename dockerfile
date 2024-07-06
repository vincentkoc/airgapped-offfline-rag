FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for models and database
RUN mkdir -p models chroma_db

# Set environment variable to use GPU if available
ENV CUDA_VISIBLE_DEVICES=all

# Run the application
CMD ["streamlit", "run", "app/main.py", "--browser.serverAddress", "localhost"]
