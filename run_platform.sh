#!/usr/bin/env bash
# The Analysis Room — one command to run the whole platform locally.
#
#   ./run_platform.sh
#
# Starts the FastAPI backend (local MLX coach) on :8000 and the Next.js front end
# on :3000, then waits. Ctrl-C stops both.
#
# HONESTY NOTE: this local runner serves whatever COACH_MODEL_PATH points at. With
# NO COACH_MODEL_PATH set (and no local tuned dir present), the backend loads the
# UNTUNED BASE model (mlx-community/Qwen3-1.7B-4bit) — it is NOT the shipped v4
# coach. The shipped v4 (Qwen3-32B QLoRA) surface is the live HF Space + the Modal
# v4 endpoint (printed in the startup banner). Point COACH_MODEL_PATH at a tuned
# MLX model dir (or set COACH_ADAPTER_PATH) to serve a tuned coach locally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# PY: honor $PY; else the Mac MLX venv if present; else system python3 (cloud VM).
if [ -z "${PY:-}" ]; then
  if [ -x "$HOME/.venvs/mlx/bin/python" ]; then PY="$HOME/.venvs/mlx/bin/python"; else PY="python3"; fi
fi
# STOCKFISH_PATH: honor $STOCKFISH_PATH; else a stockfish on PATH; else Mac brew default.
STOCKFISH_PATH="${STOCKFISH_PATH:-$(command -v stockfish || echo /opt/homebrew/bin/stockfish)}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"

# Load .env (OpenAI/TFY keys, and possibly COACH_MODEL_PATH) if present — never printed.
# Sourced BEFORE choosing the coach so an explicit COACH_MODEL_PATH in .env is honored.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# Decide WHAT coach actually runs — and be honest about it (no silent v2 default):
#   1) explicit COACH_MODEL_PATH (env/.env)      -> serve that tuned model
#   2) else a real local tuned dir if it exists  -> serve it
#   3) else leave UNSET                          -> backend loads the UNTUNED BASE
LOCAL_TUNED="models/mlx/chess-coach-v2"
LIVE_SPACE="https://khoilamalphaai-chess-coach-studio.static.hf.space"
MODAL_V4="chess-coach-v4-4bit-maia (Modal A100 vLLM, chess-instructor-3)"
if [ -n "${COACH_MODEL_PATH:-}" ]; then
  COACH_SURFACE="tuned model: ${COACH_MODEL_PATH}"
elif [ -d "$LOCAL_TUNED" ]; then
  COACH_MODEL_PATH="$LOCAL_TUNED"
  COACH_SURFACE="local tuned model: ${COACH_MODEL_PATH}"
else
  COACH_SURFACE="UNTUNED BASE (mlx-community/Qwen3-1.7B-4bit) — NOT the shipped v4 coach"
fi

# Export ONLY when set: exporting an EMPTY COACH_MODEL_PATH would make the backend
# load "" instead of falling back to its base-model default.
export STOCKFISH_PATH
if [ -n "${COACH_MODEL_PATH:-}" ]; then export COACH_MODEL_PATH; fi
if [ -n "${COACH_ADAPTER_PATH:-}" ]; then export COACH_ADAPTER_PATH; fi

echo "[run] python   : $PY"
echo "[run] backend  : ${COACH_SURFACE}  ->  http://127.0.0.1:${API_PORT}"
if [ -z "${COACH_MODEL_PATH:-}" ]; then
  echo "[run] NOTE     : no COACH_MODEL_PATH set — this serves the UNTUNED BASE locally, NOT v4."
  echo "[run]            set COACH_MODEL_PATH=/path/to/tuned-mlx (or COACH_ADAPTER_PATH) to serve a tuned coach."
  echo "[run]            the shipped v4 (Qwen3-32B QLoRA) coach is live at:"
  echo "[run]              • Space: ${LIVE_SPACE}"
  echo "[run]              • Modal: ${MODAL_V4}"
fi
echo "[run] stockfish: $STOCKFISH_PATH"
echo "[run] frontend :                    ->  http://localhost:${WEB_PORT}"

"$PY" -m uvicorn src.api.server:app --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

( cd web && WEB_PORT="$WEB_PORT" npm run dev -- --port "$WEB_PORT" ) &
WEB_PID=$!

cleanup() { echo; echo "[run] stopping…"; kill "$API_PID" "$WEB_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait
