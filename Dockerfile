FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Generate the synthetic dataset and train both models while the image is built.
# The container then starts instantly and needs no external files or credentials.
RUN python -m app.ml > /dev/null

# Run as a normal user instead of root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"

# Render (and most platforms) tell the app which port to use through the PORT variable.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
