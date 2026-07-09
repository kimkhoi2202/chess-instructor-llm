#!/usr/bin/env bash
# The Analysis Room — one command to run the whole platform locally.
#
#   ./run_platform.sh
#
# Starts the FastAPI backend (tuned MLX coach) on :8000 and the Next.js front end
# on :3000, then waits. Ctrl-C stops both. Override the model with COACH_MODEL_PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# PY: honor $PY; else the Mac MLX venv if present; else system python3 (cloud VM).
if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/.venvs/mlx/bin/python" ]; then PY="$HOME/.venvs/mlx/bin/python"; else PY="python3"; fi
fi
COACH_MODEL_PATH="${COACH_MODEL_PATH:-models/mlx/chess-coach-v2}"
# STOCKFISH_PATH: honor $STOCKFISH_PATH; else a stockfish on PATH; else Mac brew default.
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || echo /opt/homebrew/bin/stockfish)}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

# Load .env (OpenAI/TFY keys etc.) if present — never printed.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export COACH_MODEL_PATH STOCKFISH_PATH

echo "[run] python   : $PY"
echo "[run] backend  : $COACH_MODEL_PATH  ->  http://127.0.0.1:${API_PORT}"
echo "[run] stockfish: $STOCKFISH_PATH"
echo "[run] frontend :                    ->  http://localhost:${WEB_PORT}"

"$PY" -m uvicorn src.api.server:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

( cd web && WEB_PORT="$WEB_PORT" npm run dev -- --port "$WEB_PORT" ) &
WEB_PID=$!

cleanup() { echo; echo "[run] stopping…"; kill "$API_PID" "$WEB_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait
