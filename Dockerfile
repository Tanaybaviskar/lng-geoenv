# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Install dependencies using uv
RUN uv sync --frozen --no-install-project

# Set environment variable (important for imports)
ENV PYTHONPATH=/app

# Default command
CMD ["uv", "run", "python", "inference.py"]