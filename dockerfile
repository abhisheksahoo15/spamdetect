# Dockerfile
# This builds a Docker image for your FastAPI application, including NLTK data.

# Use Python 3.9 base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT: Download NLTK data during Docker build ---
# This ensures 'stopwords' and 'punkt' are available in the container
# without needing to download at runtime, preventing deployment issues.
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy application code, model, and vectorizer
COPY . .

# Expose port 80
EXPOSE 80

# Run the application using Uvicorn
# --host 0.0.0.0 is to listen on all available network interfaces
# --port 80 is to listen on port 80 inside the container
# main:app refers to the 'app' object in the main.py file
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
