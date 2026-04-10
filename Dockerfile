# HF Spaces compatible Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Combine both root and server requirements to be safe
COPY requirements.txt .
COPY server/requirements.txt server_requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r server_requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
