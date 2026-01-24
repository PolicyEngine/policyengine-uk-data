FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn sqlmodel pandas

# Copy targets module
COPY targets /app/policyengine_uk_data/targets

# Copy backend module
COPY explorer/backend /app/policyengine_uk_data/explorer/backend

# Create __init__.py files
RUN mkdir -p /app/policyengine_uk_data/explorer && \
    touch /app/policyengine_uk_data/__init__.py && \
    touch /app/policyengine_uk_data/explorer/__init__.py

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "policyengine_uk_data.explorer.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
