#!/usr/bin/env bash
# Surgery Platform Deployment Script
# Usage: ./deploy-surgery.sh [--local-llm] [--skip-build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOCAL_LLM=false
SKIP_BUILD=false

for arg in "$@"; do
    case $arg in
        --local-llm) LOCAL_LLM=true ;;
        --skip-build) SKIP_BUILD=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "═══════════════════════════════════════════════"
echo "  Surgery Annotation Platform - Deployment"
echo "═══════════════════════════════════════════════"

# ── Check prerequisites ──
echo ""
echo "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "ERROR: docker compose not found"; exit 1; }
echo "  ✓ Docker"

if nvidia-smi >/dev/null 2>&1; then
    echo "  ✓ NVIDIA GPU detected"
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "    GPU memory: ${GPU_MEM}MB"
else
    echo "  ⚠ No NVIDIA GPU detected — model services will not work"
fi

# ── Check .env ──
if [ ! -f .env ]; then
    echo ""
    echo "No .env file found. Creating from template..."
    cp .env.surgery.example .env
    echo "  ✓ Created .env — please edit with your settings"
    echo "    vim .env"
    echo ""
    echo "Then re-run: ./deploy-surgery.sh"
    exit 0
fi
echo "  ✓ .env file"

# Export .env values for compose interpolation and local health checks
set -a
. ./.env
set +a

export ALLOWED_HOSTS="${ALLOWED_HOSTS:-${CVAT_HOST:-localhost},localhost,127.0.0.1}"

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.surgery.yml"
PUBLIC_URL="${CVAT_BASE_URL:-http://localhost:8080}"
ENABLE_HTTPS=false

if [ -n "${ACME_EMAIL:-}" ]; then
    ENABLE_HTTPS=true
fi

if [ "$ENABLE_HTTPS" = true ]; then
    COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.https.yml"
    PUBLIC_URL="${CVAT_BASE_URL:-https://${CVAT_HOST}}"
elif [[ "${CVAT_BASE_URL:-}" =~ ^https:// ]]; then
    echo "  (Public HTTPS URL configured without local ACME; assuming TLS terminates upstream)"
fi

# ── Build images ──
if [ "$SKIP_BUILD" = false ]; then
    echo ""
    echo "Building Docker images..."
    docker compose -f docker-compose.yml -f docker-compose.surgery.yml build
    echo "  ✓ Images built"
fi

# ── Start services ──
echo ""
echo "Starting services..."

PROFILES=""
if [ "$LOCAL_LLM" = true ]; then
    PROFILES="--profile local-llm"
    echo "  (Including local LLM service)"
fi
if [ "$ENABLE_HTTPS" = true ]; then
    echo "  (Enabling Traefik HTTPS overlay for ${CVAT_HOST})"
fi

docker compose $COMPOSE_FILES $PROFILES up -d
echo "  ✓ Services started"

# ── Wait for database ──
echo ""
echo "Waiting for database..."
for i in $(seq 1 30); do
    if docker compose exec -T cvat_db pg_isready -U root >/dev/null 2>&1; then
        echo "  ✓ Database ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "  ✗ Database not ready after 30s"
        exit 1
    fi
    sleep 1
done

# ── Run migrations ──
echo ""
echo "Running database migrations..."
docker compose exec -T cvat_server python3 manage.py migrate
echo "  ✓ Migrations complete"

# ── Create superuser if none exists ──
echo ""
echo "Checking admin user..."
ADMIN_EXISTS=$(docker compose exec -T cvat_server python3 manage.py shell -c "
from django.contrib.auth.models import User
print(User.objects.filter(is_superuser=True).exists())
" 2>/dev/null | tail -1)

if [ "$ADMIN_EXISTS" = "False" ]; then
    echo "  Creating admin user..."
    echo "  Enter credentials for the admin account:"
    docker compose exec cvat_server python3 manage.py createsuperuser
else
    echo "  ✓ Admin user exists"
fi

# ── Health checks ──
echo ""
echo "Running health checks..."

CVAT_CHECK_OPTS=()
CVAT_API_CHECK_URL="http://localhost:8080/api/server/about"
CVAT_UI_CHECK_URL="http://localhost:8080/"
CVAT_DISPLAY_URL="${PUBLIC_URL}"

if [ "$ENABLE_HTTPS" = true ]; then
    CVAT_CHECK_OPTS=(--resolve "${CVAT_HOST}:443:127.0.0.1" -k)
    CVAT_API_CHECK_URL="https://${CVAT_HOST}/api/server/about"
    CVAT_UI_CHECK_URL="https://${CVAT_HOST}/"
fi

# CVAT server
if curl "${CVAT_CHECK_OPTS[@]}" -sf "$CVAT_API_CHECK_URL" >/dev/null 2>&1; then
    echo "  ✓ CVAT server (${CVAT_DISPLAY_URL})"
else
    echo "  ⚠ CVAT server not responding yet (may still be starting)"
fi

# CVAT UI
if curl "${CVAT_CHECK_OPTS[@]}" -sf "$CVAT_UI_CHECK_URL" >/dev/null 2>&1; then
    echo "  ✓ CVAT UI"
else
    echo "  ⚠ CVAT UI not responding yet"
fi

# Parakeet ASR
if curl -sf http://localhost:${PARAKEET_PORT:-8888}/docs >/dev/null 2>&1; then
    echo "  ✓ Parakeet ASR (http://localhost:${PARAKEET_PORT:-8888})"
else
    echo "  ⚠ Parakeet ASR not ready (model may still be loading)"
fi

# LLM
if [ "$LOCAL_LLM" = true ]; then
    if curl -sf http://localhost:${LLM_PORT:-8080}/v1/models >/dev/null 2>&1; then
        echo "  ✓ Local LLM (http://localhost:${LLM_PORT:-8080})"
    else
        echo "  ⚠ Local LLM not ready (model may still be loading)"
    fi
fi

# ── Summary ──
echo ""
echo "═══════════════════════════════════════════════"
echo "  Deployment complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Platform:  ${PUBLIC_URL}"
echo "  My Work:   ${PUBLIC_URL}/my-work"
echo "  Datasets:  ${PUBLIC_URL}/datasets"
echo "  QA:        ${PUBLIC_URL}/surgery-qa"
echo "  Ontology:  ${PUBLIC_URL}/ontology"
echo ""
echo "  Next steps:"
echo "  1. Log in with your admin credentials"
echo "  2. Create a project with surgical phase labels"
echo "  3. Register your S3 cloud storage (Settings → Cloud Storages)"
echo "  4. Go to Datasets → create a dataset or use Ingest page"
echo "  5. Assign jobs to surgeons"
echo ""
if [ "$ENABLE_HTTPS" = true ]; then
    echo "  HTTPS is enabled via Traefik + Let's Encrypt."
    echo "  Ensure ${CVAT_HOST} resolves directly to this host and ports 80/443 are open."
else
    echo "  For external access, set CVAT_HOST/CVAT_BASE_URL in .env"
    echo "  to your domain and add ACME_EMAIL for docker-compose.https.yml"
fi
echo ""
