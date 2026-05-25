#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

show_help() {
    cat <<'EOF'
Usage: ./start.sh [options] [extra docker compose up args/services...]

Options:
  --dev         Include docker-compose.dev.yml
  --https       Force-enable docker-compose.https.yml
  --no-https    Force-disable docker-compose.https.yml
  --build       Build images before starting containers
  -h, --help    Show this help

Examples:
  ./start.sh
  ./start.sh --build
  ./start.sh --dev
  ./start.sh cvat_ui
  ./start.sh --dev --build cvat_server cvat_ui
EOF
}

has_env_var() {
    local key="$1"
    [[ -f .env ]] && grep -Eq "^[[:space:]]*${key}=.+" .env
}

use_dev=false
use_https="auto"
use_build=false
extra_args=()

while (($#)); do
    case "$1" in
        --dev)
            use_dev=true
            ;;
        --https)
            use_https=true
            ;;
        --no-https)
            use_https=false
            ;;
        --build)
            use_build=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            extra_args+=("$1")
            ;;
    esac
    shift
done

compose_files=(-f docker-compose.yml)
if $use_dev; then
    compose_files+=(-f docker-compose.dev.yml)
fi
compose_files+=(-f docker-compose.surgery.yml)

if [[ "$use_https" == "auto" ]]; then
    if has_env_var "ACME_EMAIL"; then
        use_https=true
    else
        use_https=false
    fi
fi

if [[ "$use_https" == "true" ]]; then
    compose_files+=(-f docker-compose.https.yml)
fi

cmd=(docker compose "${compose_files[@]}" up -d)
if $use_build; then
    cmd+=(--build)
fi
cmd+=("${extra_args[@]}")

printf '+'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
