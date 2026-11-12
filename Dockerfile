FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY prompts/ ./prompts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV USE_CLOUD_STORAGE=true
ENV PYTHONPATH=/app/src

# Run as non-root
RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

# Production WSGI server. One worker: /run holds the request open for the whole
# generation and the duplicate-run lock in server.py is per-process.
CMD exec gunicorn --bind :$PORT --chdir /app/src \
    --workers 1 --threads 8 --worker-class gthread \
    --timeout 3600 --access-logfile - server:app
