# Use Python 3.11 slim base image for portability and efficiency
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements to leverage Docker cache
COPY requirements.txt .

# Install dependencies using PyTorch CPU wheels for portability
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code into the app workspace
COPY frontend/ ./frontend/
COPY learning_hub/ ./learning_hub/
COPY test_image.jpg ./

# Expose the Streamlit service port
EXPOSE 8501

# Configure environment variables
ENV BACKEND_URL=http://backend:8000
ENV PYTHONPATH=/app

# Startup command for Streamlit
CMD ["python", "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
