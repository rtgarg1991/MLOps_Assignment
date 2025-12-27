FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
# Copy data folder (REQUIRED for ingest.py to find source file)
COPY data/ ./data/

# Generic entrypoint allows K8s to pass "src/script_name.py"
ENTRYPOINT ["python"]