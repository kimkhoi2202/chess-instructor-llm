# The Analysis Room — chess coaching studio

A calm, board-centric web app that demos the engine-grounded chess coach. You play
the move you are unsure about, and the coach picks one leveled teaching move and
explains it in plain language — the recommendation drawn as a brass arrow, your
move as a faint rust arrow, the lesson set in a literary serif.

- **Front end** (`web/`): Next.js 16 App Router + TypeScript + Tailwind CSS v4 +
  [HeroUI v3](https://heroui.com) + [`react-chessboard`](https://react-chessboard.vercel.app) v5.
- **Back end** (`src/api/server.py`): a thin FastAPI layer that reuses the repo's
  existing `stockfish_engine`, `maia_engine`, `config/schema`, `prompts/coach_system.md`,
  and the MLX coach model. It does not re-implement any chess logic.

## Prerequisites

- The MLX venv Python with `mlx_lm`, `python-chess`, `fastapi`, `uvicorn`
  (`/Users/khoilam/.venvs/mlx/bin/python`).
- Stockfish (`/opt/homebrew/bin/stockfish`) — required.
- lc0 + the Maia nets in `models/maia/` — optional. If missing, the coach still
  runs; the human-likelihood (Maia) panel just shows "unavailable".
- Node 18.18+ (built and tested on Node 26) for the front end.

## Run (two commands)

**Terminal 1 — backend (from the repo root `chess-instructor-llm/`):**

```bash
~/.venvs/mlx/bin/python -m uvicorn src.api.server:app --port 8000
```

Loads the MLX coach once at startup and serves `GET /api/health`,
`GET /api/examples`, and `POST /api/coach` with CORS open to `http://localhost:3000`.

**Terminal 2 — front end (from `chess-instructor-llm/web/`):**

```bash
npm install   # first time only
npm run dev
```

Open http://localhost:3000. The page auto-runs the coach on the classic 2.Qh5
example so you land straight in a coaching reveal.

## Point at the fine-tuned model

The tuned-model swap is a single environment variable on the backend command —
nothing else changes:

```bash
# a fused MLX model directory or repo
COACH_MODEL_PATH=./models/qwen3-coach-mlx \
  ~/.venvs/mlx/bin/python -m uvicorn src.api.server:app --port 8000

# ...or keep the base weights and apply an MLX LoRA adapter
COACH_ADAPTER_PATH=/path/to/mlx-adapter \
  ~/.venvs/mlx/bin/python -m uvicorn src.api.server:app --port 8000
```

When either is set, `/api/health` reports `"tuned": true` and the UI badge reads
"Tuned coach" instead of "Base model".

The front end targets the backend via `NEXT_PUBLIC_API_BASE` (see `.env.local`,
default `http://127.0.0.1:8000`).

## Production build

```bash
npm run build   # from web/ — type-checks and compiles
npm run start   # serve the production build
```

## Setup notes / gotchas

- **HeroUI v3** requires **Tailwind CSS v4** and **React 19**, and needs **no
  provider**. It is wired via two lines in `src/app/globals.css`
  (`@import "tailwindcss"; @import "@heroui/styles";`) and themed entirely through
  OKLCH CSS variables (the Analysis Room palette overrides HeroUI's semantic
  tokens in the same file). v3 uses compound components (`Card.Header`,
  `ToggleButtonGroup` + `ToggleButton`, `Chip.Label`, `Tooltip.Content`).
- **react-chessboard v5** takes a single `options` prop (`position`, `arrows`,
  `darkSquareStyle` / `lightSquareStyle`, `boardOrientation`, `onPieceDrop`, ...).
  It is dynamically imported with `ssr: false`. The coaching arrows are a custom
  animated SVG overlay (viewBox `0 0 8 8`) so the brass "draw" is fully controlled
  and respects `prefers-reduced-motion`.
- The board is an **annotation surface**: dragging a piece marks "the move you are
  unsure about" (drawn as a rust arrow) without changing the position — the FEN
  stays fixed so the backend gets `{position, student_move}`.
- The coach reply is free text; the backend reuses `render_user_prompt` and appends
  a small format hint so the reply parses into a coaching body plus one
  `Takeaway:` line for the pull-quote. The recommended move is always a sound,
  non-student move (it falls back to the engine's best sound move if the prose
  names the piece instead of the SAN).
