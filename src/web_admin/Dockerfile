FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY src/web_admin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src modules we need
COPY src/config.py ./config.py
COPY src/storage_adapter.py ./storage_adapter.py
COPY src/topic_creator_schemas.py ./topic_creator_schemas.py
COPY src/topic_manager.py ./topic_manager.py
COPY src/smart_suggester.py ./smart_suggester.py
COPY src/llm_client.py ./llm_client.py

# Copy web admin application code
COPY src/web_admin/ ./

# Copy prompts
COPY prompts/ ./prompts/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PYTHONPATH=/app

# Run as non-root
RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

# Production WSGI server. Never the Flask dev server - it would expose the
# Werkzeug debugger. Timeout stays under the Cloud Run request timeout (300s).
CMD exec gunicorn --bind :$PORT --chdir /app \
    --workers 1 --threads 8 --worker-class gthread \
    --timeout 320 --access-logfile - app:app
