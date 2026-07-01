#!/usr/bin/env bash
# Bring up the whole local stack with one command: Redis Stack (checkpointer +
# retrieval cache), then the FastAPI app with the static frontend mounted on
# the same origin (RAG/app.py via _dev_ui.py). Qdrant is file-based locally
# (RAG/vector_db/qdrant) and needs no separate process.
#
# Usage:
#   ./start.sh          start everything (idempotent — safe to re-run)
#   ./start.sh stop      stop the app and the local Redis container
#   ./start.sh status     print current health without starting anything
#   ./start.sh logs       tail the app log
#
# Env overrides: PORT (default 8080), REDIS_CONTAINER (default rag-redis-6380)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .venv and .env are not part of a git worktree checkout — they live at the
# main repo root. Resolve it via git-common-dir so this script works whether
# it's run from the main checkout or from a worktree under it.
MAIN_ROOT="$(dirname "$(git rev-parse --git-common-dir 2>/dev/null || echo "$SCRIPT_DIR/.git")")"
[ -d "$MAIN_ROOT/.venv" ] || MAIN_ROOT="$SCRIPT_DIR"

VENV="$MAIN_ROOT/.venv"
ENV_FILE="$MAIN_ROOT/.env"
PORT="${PORT:-8080}"
REDIS_CONTAINER="${REDIS_CONTAINER:-rag-redis-6380}"
LOG_FILE="/tmp/rag-langchain-dev.log"
PID_FILE="/tmp/rag-langchain-dev.pid"

log() { echo "==> $*"; }
warn() { echo "!!  $*" >&2; }

app_status_json() {
  curl -fsS "http://localhost:${PORT}/status" 2>/dev/null || true
}

app_is_up() {
  local body
  body="$(app_status_json)"
  [ -n "$body" ] && echo "$body" | grep -q '"phase"'
}

redis_is_up() {
  docker exec "$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1
}

do_stop() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    log "Stopping app (pid $(cat "$PID_FILE"))"
    kill "$(cat "$PID_FILE")"
  else
    warn "No tracked app process; if one is bound to :$PORT stop it manually."
  fi
  rm -f "$PID_FILE"
  # Redis is left running: it's a shared container that other local instances
  # of this app (e.g. one started from a different checkout) may depend on.
  # Stop it explicitly if needed: docker stop "$REDIS_CONTAINER"
}

do_status() {
  if redis_is_up; then log "Redis: up ($REDIS_CONTAINER)"; else warn "Redis: down"; fi
  if app_is_up; then
    log "App: up — http://localhost:${PORT}"
    app_status_json
  else
    warn "App: down"
  fi
}

case "${1:-up}" in
  stop) do_stop; exit 0 ;;
  status) do_status; exit 0 ;;
  logs) exec tail -f "$LOG_FILE" ;;
  up) ;;
  *) echo "Usage: $0 [up|stop|status|logs]"; exit 1 ;;
esac

log "Repo:  $SCRIPT_DIR"
log "Env:   $ENV_FILE"
log "Venv:  $VENV"

# ── 1. Python environment ────────────────────────────────────────────────────
# The shared main-checkout venv can diverge from this branch's pyproject.toml
# (e.g. it's synced against an in-progress, uncommitted dependency change on
# another branch) — qdrant-client missing is exactly that. Detect it with a
# real import rather than trusting the venv exists, and fall back to a venv
# scoped to this checkout instead of mutating the shared one.
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import qdrant_client" >/dev/null 2>&1; then
  if [ -x "$VENV/bin/python" ]; then
    warn "Shared venv at $VENV doesn't satisfy this checkout's dependencies (e.g. qdrant-client) — using an isolated venv here instead."
  else
    log "No .venv found — running 'uv sync --dev'"
  fi
  command -v uv >/dev/null || { warn "uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
  uv sync --dev
  VENV="$SCRIPT_DIR/.venv"
fi

# ── 2. Environment variables ─────────────────────────────────────────────────
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  warn "No .env found at $ENV_FILE — continuing with defaults."
fi

# auth.py requires a valid bearer JWT whenever JWT_SECRET is set, and this
# codebase has no /login route to mint one — the frontend never sends an
# Authorization header. _dev_ui.py's own ALLOW_INSECURE_DEV=1 default only
# takes effect when JWT_SECRET is empty, so a local run must not export it.
unset JWT_SECRET
export ALLOW_INSECURE_DEV=1

# This repo's .env carries settings for a provider-agnostic embedding stack
# (e.g. EMBEDDING_MODEL_NAME="gemini-embedding-2-preview") that this checkout's
# RAG/services/embeddings.py does not implement — it passes the value straight
# to HuggingFaceEmbeddings, which then hangs retrying a nonexistent HF repo and
# /status never leaves "loading_models". Unset it so the code's own working
# default (sentence-transformers/all-MiniLM-L6-v2) is used.
if [ -n "${EMBEDDING_MODEL_NAME:-}" ] && [ "$EMBEDDING_MODEL_NAME" != "sentence-transformers/all-MiniLM-L6-v2" ]; then
  warn "Ignoring EMBEDDING_MODEL_NAME='$EMBEDDING_MODEL_NAME' — not a HuggingFace repo id this codebase understands."
  unset EMBEDDING_MODEL_NAME
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  warn "OPENROUTER_API_KEY is not set — /ask will fail at the LLM call."
  warn "Retrieval, upload, and admin endpoints still work without it."
elif [[ "${LLM_MODEL:-}" != */* ]]; then
  warn "LLM_MODEL='${LLM_MODEL:-}' has no 'vendor/model' prefix — OpenRouter will likely 404 on it."
fi

# This machine's Python trust store rejects huggingface.co's cert chain
# ("self-signed certificate in certificate chain") even though the models are
# already downloaded — so force the HF stack to use the local cache instead
# of validating freshness over the network. Falls back to online mode (and a
# warning) if a required model isn't cached yet, e.g. on a fresh machine.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
_hf_cached() {
  local repo_dir="models--${1//\//--}"
  [ -d "$HF_HOME/hub/$repo_dir" ]
}
_embedding_model="${EMBEDDING_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
_reranker_model="${RERANKER_MODEL_NAME:-BAAI/bge-reranker-base}"
if _hf_cached "$_embedding_model" && { [ "${RAG_ENABLE_RERANKER:-1}" = "0" ] || _hf_cached "$_reranker_model"; }; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  warn "Required HF model(s) not found in $HF_HOME/hub — first run needs network access to huggingface.co."
fi

# ── 3. Redis Stack (RediSearch is required for the LangGraph checkpointer) ──
command -v docker >/dev/null || { warn "docker not found — required to run Redis Stack locally."; exit 1; }
docker info >/dev/null 2>&1 || { warn "Docker daemon not running — start Docker Desktop first."; exit 1; }

REDIS_PORT="$("$VENV/bin/python" -c "
from urllib.parse import urlparse
import os
print(urlparse(os.environ.get('REDIS_URL', 'redis://localhost:6380')).port or 6380)
")"

if redis_is_up; then
  log "Redis Stack already running ($REDIS_CONTAINER)"
elif docker ps -a --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
  log "Starting existing container $REDIS_CONTAINER"
  docker start "$REDIS_CONTAINER" >/dev/null
else
  log "Creating Redis Stack container $REDIS_CONTAINER on port $REDIS_PORT"
  docker run -d --name "$REDIS_CONTAINER" -p "${REDIS_PORT}:6379" redis:8 >/dev/null
fi

for _ in $(seq 1 15); do
  redis_is_up && break
  sleep 1
done
redis_is_up || { warn "Redis did not become ready in time."; exit 1; }

# ── 4. App (frontend + API on one origin) ───────────────────────────────────
if app_is_up; then
  log "App already running at http://localhost:${PORT}"
else
  log "Starting app on http://localhost:${PORT} (log: $LOG_FILE)"
  nohup "$VENV/bin/python" -m uvicorn _dev_ui:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  printf "==> Waiting for models/index to warm up"
  ready=0
  for _ in $(seq 1 150); do
    body="$(app_status_json)"
    if echo "$body" | grep -q '"phase":"ready"'; then
      ready=1
      break
    fi
    printf "."
    sleep 2
  done
  echo
  if [ "$ready" -ne 1 ]; then
    warn "Still warming up after 5 minutes — check $LOG_FILE"
  fi
fi

echo
log "RAG Multi-Agent System: http://localhost:${PORT}"
log "Logs:   ./start.sh logs"
log "Stop:   ./start.sh stop"
[ -d "$MAIN_ROOT/RAG/data/workspace/markdown" ] || warn "No documents indexed yet — upload one from the UI (admin) to populate the knowledge base."
