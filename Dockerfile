# Start with Python
FROM python:3.10-slim

# Install the Operating System dependencies (Tesseract & Poppler)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set up the folder
WORKDIR /app

# Copy your files
COPY requirements.txt .
COPY main.py .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Start the server (Railway assigns a random port to $PORT, we must use it)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
