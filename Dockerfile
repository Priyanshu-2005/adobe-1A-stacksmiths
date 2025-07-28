# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app code (including onnx_model)
COPY app/ ./app/

# Move into the app directory and link entrypoint
WORKDIR /app/app
CMD ["python", "main.py"]
