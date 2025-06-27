FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Faster layer caching: copy only requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last
COPY app ./app

# Expose service port
EXPOSE 443

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "443"]
