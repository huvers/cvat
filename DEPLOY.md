# Surgery Annotation Platform — Deployment Guide

## Prerequisites

- Linux workstation with NVIDIA GPU (>= 8GB VRAM recommended)
- Docker + Docker Compose v2
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- AWS credentials for S3 bucket access

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/huvers/cvat.git
cd cvat
git checkout develop

# 2. Create your config
cp .env.surgery.example .env
vim .env   # Edit LLM settings (see options below)

# 3. Deploy
./deploy-surgery.sh

# 4. Open http://localhost:8080
```

## LLM Configuration Options

### Option A: Local LLM (llama-cpp-python)

Best for air-gapped environments or when you want full control.

```bash
# Download a GGUF model
mkdir -p models
wget -O models/model.gguf \
  "https://huggingface.co/TheBloke/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"

# Edit .env
LLM_API_URL=http://llm:8000/v1/chat/completions
LLM_MODEL=default
LLM_MODEL_FILE=model.gguf

# Deploy with local LLM
./deploy-surgery.sh --local-llm
```

### Option B: Remote API (OpenAI-compatible)

Best for quality and simplicity. Works with OpenAI, Anthropic, Together, etc.

```bash
# Edit .env
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_MODEL=gpt-4o

# Set your API key
export OPENAI_API_KEY=sk-...

# Deploy without local LLM
./deploy-surgery.sh
```

### Option C: NVIDIA Inference API for Copilot

Best when you want Copilot to use NVIDIA's hosted OpenAI-compatible endpoint.

```bash
# Edit .env
COPILOT_LLM_URL=https://inference-api.nvidia.com/v1/chat/completions
COPILOT_LLM_MODEL=azure/openai/gpt-5.4

# Set your API key
NVIDIA_API_KEY=nvapi-...

# Deploy
./deploy-surgery.sh
```

## First-Time Setup

### 1. Create a Project

- Log in at http://localhost:8080 with admin credentials
- Go to Projects → Create
- Add labels for your surgical phases (e.g., "Preparation", "Calot Triangle Dissection", "Clipping & Cutting", etc.)
- Add labels for anatomy (for SAM3): "gallbladder", "cystic duct", "liver", etc.
- Add classification labels: "Cholecystectomy", "Hernia Repair", etc.

### 2. Register S3 Cloud Storage

- Go to Cloud Storages → Create
- Provider: AWS S3
- Bucket name: your-bucket
- Credentials: Access key + Secret key
- Region: your-region

### 3. Create a Dataset

- Go to Datasets page (or use API)
- The first bulk ingest will auto-create a Dataset

### 4. Ingest Videos

- Go to Ingest page: http://localhost:8080/bulk-ingest
- Enter: Cloud Storage ID, procedure prefix (e.g., "cholecystectomy"), Project ID
- Click "Start Ingestion"
- Or use the Datasets page: click Sync then Ingest

### 5. Register AI Models (optional)

```bash
# Register a temporal model
curl -X POST http://localhost:8080/api/surgery-models/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cholec-phase-v2",
    "procedure_type": "Cholecystectomy",
    "model_type": "phase_classifier",
    "endpoint_url": "http://your-model-service:8080/predict"
  }'

# Register a SAM3 model
curl -X POST http://localhost:8080/api/surgery-models/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sam3-cholec",
    "procedure_type": "Cholecystectomy",
    "model_type": "anatomy_segmenter",
    "endpoint_url": "http://your-sam3-service:8080/predict"
  }'
```

### 6. Assign Jobs to Surgeons

- Create user accounts for each surgeon
- Assign jobs via the CVAT UI or API
- Surgeons navigate to http://localhost:8080/my-work

## External Access

For testing with external users:

```bash
# Option A: Cloudflare Tunnel (simplest)
cloudflared tunnel --url http://localhost:8080

# Option B: HTTPS with Let's Encrypt
# Edit .env:
CVAT_HOST=your-domain.com
# Then:
docker compose -f docker-compose.yml -f docker-compose.surgery.yml \
  -f docker-compose.https.yml up -d
```

## Architecture

```
                    ┌──────────┐
                    │  Traefik │ :8080
                    │  (proxy) │
                    └────┬─────┘
                 ┌───────┼───────┐
                 ▼       ▼       ▼
           ┌─────────┐ ┌────┐ ┌──────────┐
           │  Server  │ │ UI │ │  Workers │
           │ (Django) │ │    │ │ (RQ)     │
           └────┬─────┘ └────┘ └────┬─────┘
                │                    │
         ┌──────┼──────┐      ┌─────┼─────┐
         ▼      ▼      ▼      ▼     ▼     ▼
      ┌────┐ ┌─────┐ ┌───┐ ┌─────┐ ┌───┐ ┌─────────┐
      │ DB │ │Redis│ │OPA│ │Parakeet│LLM│ │ Model   │
      │    │ │     │ │   │ │ (ASR) │   │ │Services │
      └────┘ └─────┘ └───┘ └──GPU──┘└───┘ └───GPU───┘
```

## Monitoring

- Django RQ dashboard: http://localhost:8080/django-rq
- Server logs: `docker compose logs -f cvat_server`
- Worker logs: `docker compose logs -f cvat_worker_annotation`
- ASR logs: `docker compose logs -f parakeet`
