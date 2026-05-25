FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY components/sam3/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt \
        --extra-index-url=https://download.pytorch.org/whl/cu124

COPY components/sam3 /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
