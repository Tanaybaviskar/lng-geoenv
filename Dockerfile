# HF Spaces compatible Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Install dependencies AND project
RUN uv sync --frozen

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=7860

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Start the server with uvicorn on port 7860
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
