# 1. Start from a small official Python image
FROM python:3.11-slim

# 2. All following commands run inside /app in the container
WORKDIR /app

# 3. Install dependencies first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the source code
COPY src/ src/

# 5. Train the model at build time so the image ships with its model
RUN python -m src.train

# 6. Document the port and start the API
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
