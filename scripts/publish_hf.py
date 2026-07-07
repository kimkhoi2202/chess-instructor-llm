"""Publish the chess-coach dataset + model to the Hugging Face Hub.

The HF MCP plugin is read-only, so publishing goes through ``huggingface_hub``
with a **Write** token (created at https://huggingface.co/settings/tokens under
the account this runs as).

Usage::

    HF_TOKEN=hf_xxx ~/.venvs/mlx/bin/python -m scripts.publish_hf
    # or
    ~/.venvs/mlx/bin/python -m scripts.publish_hf --token hf_xxx

    # optional overrides
    ... --dataset-repo chess-coach-move-review --model-repo qwen3-1.7b-chess-coach-mlx --private

It is idempotent (repos are created with ``exist_ok=True``) so re-running just
updates the files.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = _ROOT / "data" / "dataset"
MODEL_DIR = _ROOT / "models" / "mlx" / "chess-coach-v1"
BASE_MODEL = "Qwen/Qwen3-1.7B"

BEHAVIOR_SPEC = (
    "Given a position, the student's rating tier, the move the student played, and "
    "full-strength engine analysis (a sound-move pool with evals + short lines) plus "
    "the tier's human-move likelihoods, the coach recommends exactly ONE move drawn "
    "from the sound pool whose idea is explainable using only concepts appropriate to "
    "that tier, explains it in plain human terms tied to a concrete plan and to the "
    "student's actual mistake, and NEVER states raw engine numbers/centipawns, cites "
    "lines deeper than the tier's ply cap, recommends a blunder, or fabricates a tactic "
    "absent from the analysis. Every response ends with one transferable takeaway, and "
    "the same position yields simpler ideas for Beginner than for Advanced."
)


def _dataset_card(n_train: int, n_valid: int) -> str:
    return f"""---
license: cc-by-nc-4.0
task_categories:
- text-generation
language:
- en
tags:
- chess
- coaching
- distillation
- sft
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# Chess Coach — Move-Review SFT Dataset

Supervised fine-tuning data that instills **one behavior**: engine-grounded,
**rating-calibrated** chess move-review coaching that never leaks engine jargon.

> **Behavior spec.** {BEHAVIOR_SPEC}

## How it was made (the dataset *is* the deliverable)

1. **Positions** sampled from real Lichess games across three rating tiers
   (beginner 1000–1200, intermediate ~1500, advanced 1700–2000), each with the
   move a human actually played.
2. **Grounding** — every position analyzed with **Stockfish** (a tolerance-gated
   *sound-move pool* + mistake severity) and **Maia** (human-move likelihoods at
   the tier).
3. **Teacher** — **GPT-5.5** turns that structured analysis into leveled coaching
   that obeys the behavior spec (recommend one *instructive* sound move, plain
   language, one takeaway).
4. **Hard filter** — candidates are rejected unless the recommended move is in the
   sound pool, no engine numbers leak, the ply cap holds, and the format is valid.

Each row is a chat triple (`system` / `user` / `assistant`). The `user` message
is the engine-grounded prompt; the `assistant` message is the coaching label.

## Splits

| Split | Rows |
|---|---:|
| train | {n_train} |
| validation | {n_valid} |

```python
from datasets import load_dataset
ds = load_dataset("{{repo_id}}")
```

## Honest limitation

The teacher labels were filtered for *format/soundness* but **not faithfulness**,
so some explanations reference board facts that aren't literally present. A model
trained on this data reproduces the style reliably but inherits that occasional
fabrication — truthfulness needs a non-LLM verifier, not more fine-tuning. See the
companion model card for the measured base-vs-tuned eval.

## License / provenance

Positions derive from public Lichess games; coaching text is distilled from
GPT-5.5. Released for **research/education** under CC-BY-NC-4.0. Respect the
source models' terms.
"""


def _model_card(ds_repo: str) -> str:
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: mlx
pipeline_tag: text-generation
tags:
- chess
- coaching
- mlx
- qwen3
- lora
- sft
---

# Qwen3-1.7B Chess Coach (MLX, 4-bit)

A small, local specialist: **rating-calibrated chess move-review coaching**.
QLoRA SFT on [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}), merged and
quantized to **4-bit MLX** for on-device inference on Apple Silicon.

> **Behavior spec.** {BEHAVIOR_SPEC}

## Why fine-tune (not prompt)?

The target behavior **resists prompting** — the base model leaks engine numbers,
mis-levels its answers, and drifts. Fine-tuning on a controlled dataset makes the
behavior *reliable*. Measured base-vs-tuned on held-out positions, judged by a
**cross-family** LLM (Claude Opus, teacher was GPT-5.5):

| Metric | Base | Tuned |
|---|---:|---:|
| Move sound (%) | 87 | **100** |
| No engine-speak (%) | 33 | **100** |
| Spec adherence (0–2) | 0.47 | **0.93** |
| Level calibration (0–2) | 0.60 | **1.13** |
| Truthfulness (0–2) | 0.13 | 0.13 |

The fine-tune is a reliable **style/behavior compressor**. It does **not** fix
*truthfulness* (the model still sometimes fabricates board facts) — that needs a
non-LLM faithfulness verifier, which is the honest, measured finding of this project.

## Usage (MLX)

```python
from mlx_lm import load, generate
model, tok = load("{{repo_id}}")
messages = [
    {{"role": "system", "content": "You are a chess coach doing move review..."}},
    {{"role": "user", "content": "Student tier: beginner ... (engine-grounded prompt)"}},
]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tok, prompt=prompt, max_tokens=512))
```

## Training data

[`{ds_repo}`](https://huggingface.co/datasets/{ds_repo}) — Lichess positions →
Stockfish/Maia grounding → GPT-5.5 teacher → hard filter.
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    p.add_argument("--dataset-repo", default="chess-coach-move-review")
    p.add_argument("--model-repo", default="qwen3-1.7b-chess-coach-mlx")
    p.add_argument("--private", action="store_true", help="create the repos private")
    p.add_argument("--skip-model", action="store_true", help="publish only the dataset")
    p.add_argument("--skip-dataset", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, get_token

    # Prefer an explicit token/env, else fall back to a cached CLI login
    # (`hf auth login`) so the secret never has to be pasted into chat.
    token = args.token or get_token()
    if not token:
        print(
            "ERROR: no token. Either run `hf auth login` (paste your WRITE token in "
            "your own terminal), or pass --token hf_xxx / set HF_TOKEN.",
            file=sys.stderr,
        )
        return 2
    args.token = token

    api = HfApi(token=args.token)
    who = api.whoami()
    ns = who["name"]
    print(f"[hf] authenticated as {ns} (type={who.get('type')})")

    ds_repo = f"{ns}/{args.dataset_repo}"
    mdl_repo = f"{ns}/{args.model_repo}"

    # ---- Dataset ---------------------------------------------------------- #
    if not args.skip_dataset:
        train = DATASET_DIR / "train.jsonl"
        valid = DATASET_DIR / "valid.jsonl"
        n_train = sum(1 for _ in train.open())
        n_valid = sum(1 for _ in valid.open())
        print(f"[hf] dataset {ds_repo}: {n_train} train / {n_valid} valid")
        api.create_repo(ds_repo, repo_type="dataset", exist_ok=True, private=args.private)
        api.upload_file(path_or_fileobj=str(train), path_in_repo="train.jsonl",
                        repo_id=ds_repo, repo_type="dataset")
        # HF auto-detects the "validation" split from the filename.
        api.upload_file(path_or_fileobj=str(valid), path_in_repo="validation.jsonl",
                        repo_id=ds_repo, repo_type="dataset")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(_dataset_card(n_train, n_valid).replace("{repo_id}", ds_repo))
            card = fh.name
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                        repo_id=ds_repo, repo_type="dataset")
        print(f"[hf] dataset published → https://huggingface.co/datasets/{ds_repo}")

    # ---- Model ------------------------------------------------------------ #
    if not args.skip_model:
        if not (MODEL_DIR / "model.safetensors").exists():
            print(f"ERROR: model weights not found in {MODEL_DIR}", file=sys.stderr)
            return 3
        print(f"[hf] model {mdl_repo}: uploading {MODEL_DIR} (~0.9 GB)")
        api.create_repo(mdl_repo, repo_type="model", exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(MODEL_DIR), repo_id=mdl_repo,
                          ignore_patterns=["README.md", ".gitattributes"])
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(_model_card(ds_repo).replace("{repo_id}", mdl_repo))
            card = fh.name
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=mdl_repo)
        print(f"[hf] model published → https://huggingface.co/{mdl_repo}")

    print("[hf] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
