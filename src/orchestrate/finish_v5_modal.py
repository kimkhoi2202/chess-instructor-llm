#!/usr/bin/env python3
"""FINISH the v5 (32B) Stage-2 eval ENTIRELY on Modal — Mac-independent.

A single detached controller that survives a laptop shutdown: it waits for the v5
QLoRA training to finish (by polling the shared Volume for the FINAL adapter),
finishes the held-out VAL generations, runs the blinded frontier council over
TrueFoundry, computes the deterministic metrics + the v5-vs-v4-vs-4B verdict,
rebuilds the platform ``showcase.json`` seeded with v5 (filtered to distinct-tier
AND beats-frontier), and pushes the verdict report + the rebuilt showcase +
the v5 adapter to Hugging Face AND a durable Modal results Volume — so the whole
Stage-2 completes and its results survive even with the Mac off.

Nothing here is rewritten from scratch: the GPU val-gen REUSES the deployed
``chess-coach-eval-v5`` app's ``generate`` (``src/eval/eval_modal_v5.py``), and
the council / report / showcase phases REUSE ``scripts.honest_v5`` (which itself
reuses ``scripts.honest_v4``) verbatim — this module is only the Mac-independent
glue + wait-trigger + HF/volume persistence around them.

Topology (all on the SAME workspace as training + the shared adapter volume,
i.e. ``chess-instructor-4``):

* Volume ``chess-coach-lora`` (``/vol``)   — v5 adapter (``/chess-coach-v5/adapter``)
  + the val-gen output; the "training done" signal is the final
  ``/chess-coach-v5/adapter/adapter_model.safetensors``.
* Volume ``chess-coach-v5-results`` (``/results``) — durable copy of the verdict
  bundle + reseeded showcase + adapter (survives the Mac off).
* Secrets ``chess-tfy`` (TFY_API_KEY / TFY_BASE_URL / model aliases) and
  ``chess-hf`` (HF_TOKEN) — created cloud-side so the council + HF push work
  without the Mac's ``.env``.

Deploy + arm (ALWAYS scrub tokens + pin the workspace)::

    unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
    export MODAL_PROFILE=chess-instructor-4
    P=/Users/khoilam/.venvs/mlx/bin/modal
    $P deploy src/eval/eval_modal_v5.py            # bakes the 360 val prompts
    $P deploy src/orchestrate/finish_v5_modal.py   # deploys the controller
    $P run    src/orchestrate/finish_v5_modal.py::check   # dry-run the trigger logic
    $P run    src/orchestrate/finish_v5_modal.py::arm     # spawn the detached controller
    $P run    src/orchestrate/finish_v5_modal.py::status  # inspect progress later

Prereq (local, at deploy time): ``data/benchmark_v5/prompts_v5_val.jsonl`` and the
v4-field val gens under ``data/benchmark_honest/gen`` + ``data/benchmark_gap803/
scenarios.jsonl`` (all already present from the v4 eval); the images bake them so
the run needs NO Mac afterward.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import modal

# --------------------------------------------------------------------------- #
# Names / paths
# --------------------------------------------------------------------------- #
APP_NAME = "chess-coach-finish-v5"

LORA_VOLUME = "chess-coach-lora"          # shared with training (has the adapter)
RESULTS_VOLUME = "chess-coach-v5-results"  # durable results (survives Mac off)
VOL_MOUNT = "/vol"
RESULTS_MOUNT = "/results"

RUN_NAME = "chess-coach-v5"
ADAPTER_DIR = f"{VOL_MOUNT}/{RUN_NAME}/adapter"
ADAPTER_FILE = f"{ADAPTER_DIR}/adapter_model.safetensors"
GEN_REMOTE = f"{VOL_MOUNT}/{RUN_NAME}/ours_v5_val_gen.jsonl"

# The deployed val-gen app we REUSE (src/eval/eval_modal_v5.py).
EVAL_APP = "chess-coach-eval-v5"
EVAL_FN = "generate"

# HF landing spots (the durable, Mac-independent copies).
HF_ADAPTER_REPO = "khoilamalphaai/chess-coach-32b-v5-adapter"   # model repo
HF_EVAL_REPO = "khoilamalphaai/chess-coach-v5-eval"             # dataset repo

ROOT_REMOTE = "/root"
N_VAL_SCENARIOS = 360  # 120 held-out positions x 3 tiers

# Artifacts honest_v5 writes (relative to the baked repo root /root).
MD_OUT = f"{ROOT_REMOTE}/RESULTS_HONEST_EVAL_V5.md"
REPORT_JSON = f"{ROOT_REMOTE}/data/benchmark_honest/report_v5.json"
COUNCIL_JSONL = f"{ROOT_REMOTE}/data/benchmark_honest/council_v5.jsonl"
OURS_V5_LOCAL = f"{ROOT_REMOTE}/data/benchmark_v5/gen/ours_v5.jsonl"
SHOWCASE_JSON = f"{ROOT_REMOTE}/web/public/showcase.json"
SHOWCASE_STATS = f"{ROOT_REMOTE}/data/showcase/showcase_v4_stats.json"  # honest_v4 path

# Durable state on the results volume (idempotency + inspection).
STATE_PATH = f"{RESULTS_MOUNT}/v5/state.json"
LOCK_PATH = f"{RESULTS_MOUNT}/v5/_finish.lock"
DONE_PATH = f"{RESULTS_MOUNT}/v5/DONE"
BUNDLE_DIR = f"{RESULTS_MOUNT}/v5/bundle"
RESULTS_ADAPTER = f"{RESULTS_MOUNT}/v5/adapter"

TIMEOUT_S = 12 * 3600
DEFAULT_MAX_WAIT_S = 8 * 3600
POLL_INTERVAL_S = 120
LOCK_STALE_S = 11 * 3600

_PY_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.DS_Store"]

# --------------------------------------------------------------------------- #
# CPU image — python-chess + openai + HF; the repo + the data the council /
# report / showcase read are BAKED at deploy time (Mac on now) so the detached
# run needs no Mac. No GPU / no engine binaries: council + metrics are pure
# python-chess over PRE-COMPUTED scenario facts; the val-gen GPU work is reused
# from the deployed eval app.
# --------------------------------------------------------------------------- #
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
else:
    REPO = None

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("chess", "openai", "python-dotenv", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": ROOT_REMOTE})
    .workdir(ROOT_REMOTE)
)
if modal.is_local():
    cpu_image = (
        cpu_image
        .add_local_dir((REPO / "src").as_posix(), f"{ROOT_REMOTE}/src", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "config").as_posix(), f"{ROOT_REMOTE}/config", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "scripts").as_posix(), f"{ROOT_REMOTE}/scripts", copy=True, ignore=_PY_IGNORE)
        .add_local_dir((REPO / "prompts").as_posix(), f"{ROOT_REMOTE}/prompts", copy=True, ignore=_PY_IGNORE)
        # data the reused honest_v5/honest_v4 phases read (kept lean — the big
        # gap803 generations/objective files are NOT needed):
        .add_local_file((REPO / "data" / "benchmark_honest" / "val_ids.txt").as_posix(),
                        f"{ROOT_REMOTE}/data/benchmark_honest/val_ids.txt", copy=True)
        .add_local_dir((REPO / "data" / "benchmark_honest" / "gen").as_posix(),
                       f"{ROOT_REMOTE}/data/benchmark_honest/gen", copy=True, ignore=_PY_IGNORE)
        .add_local_file((REPO / "data" / "benchmark_gap803" / "scenarios.jsonl").as_posix(),
                        f"{ROOT_REMOTE}/data/benchmark_gap803/scenarios.jsonl", copy=True)
        .add_local_file((REPO / "data" / "benchmark_v5" / "prompts_v5_val.jsonl").as_posix(),
                        f"{ROOT_REMOTE}/data/benchmark_v5/prompts_v5_val.jsonl", copy=True)
        .add_local_file((REPO / "web" / "public" / "showcase.json").as_posix(),
                        f"{ROOT_REMOTE}/web/public/showcase.json", copy=True)
    )

lora_vol = modal.Volume.from_name(LORA_VOLUME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
tfy_secret = modal.Secret.from_name("chess-tfy")
hf_secret = modal.Secret.from_name("chess-hf")

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Small helpers (pure; run in-container)
# --------------------------------------------------------------------------- #
def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _gen_count() -> int:
    p = Path(GEN_REMOTE)
    if not p.exists():
        return 0
    n = 0
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _write_state(state: Dict[str, Any]) -> None:
    state["updated_ts"] = _now()
    p = Path(STATE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    results_vol.commit()


def _read_state() -> Dict[str, Any]:
    p = Path(STATE_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


# --------------------------------------------------------------------------- #
# Preflight — dry-run the trigger logic WITHOUT waiting (verification only)
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=600, volumes={VOL_MOUNT: lora_vol, RESULTS_MOUNT: results_vol},
              secrets=[tfy_secret, hf_secret])
def preflight() -> Dict[str, Any]:
    """Report every readiness signal the controller depends on (no side effects)."""
    lora_vol.reload()
    results_vol.reload()

    out: Dict[str, Any] = {
        "ts": _now(),
        "adapter_present": Path(ADAPTER_FILE).exists(),
        "adapter_dir_listing": sorted(os.listdir(ADAPTER_DIR))[:12] if Path(ADAPTER_DIR).exists() else [],
        "val_gen_rows": _gen_count(),
        "val_gen_target": N_VAL_SCENARIOS,
        "tfy_env_present": bool(os.environ.get("TFY_API_KEY")) and bool(os.environ.get("TFY_BASE_URL")),
        "hf_env_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "already_done": Path(DONE_PATH).exists(),
        "state": _read_state(),
    }

    # Can we resolve the deployed val-gen function we reuse?
    try:
        modal.Function.from_name(EVAL_APP, EVAL_FN)
        out["eval_generate_resolves"] = True
    except Exception as exc:  # noqa: BLE001
        out["eval_generate_resolves"] = False
        out["eval_generate_error"] = f"{type(exc).__name__}: {exc}"

    # Can the baked repo + data import + build the val scenarios (council input)?
    try:
        sys.path.insert(0, ROOT_REMOTE)
        os.chdir(ROOT_REMOTE)
        os.environ.setdefault("BENCH_DIR", f"{ROOT_REMOTE}/data/benchmark_gap803")
        import scripts.honest_v5 as HV5  # noqa: F401  (re-points honest_v4 globals)
        import scripts.honest_v4 as H
        scns = H._val_scenarios()
        field_present = [m for m in H.V4_FIELD if Path(f"{ROOT_REMOTE}/data/benchmark_honest/gen/{m}.jsonl").exists()]
        out["repo_import_ok"] = True
        out["val_scenarios"] = len(scns)
        out["v5_field"] = list(H.V4_FIELD)
        out["v5_field_present_excluding_ours_v5"] = field_present
        out["showcase_ours"] = H.SHOWCASE_OURS
    except Exception as exc:  # noqa: BLE001
        out["repo_import_ok"] = False
        out["repo_import_error"] = f"{type(exc).__name__}: {exc}"

    out["will_fire_when"] = (
        "adapter_model.safetensors appears on the chess-coach-lora volume at "
        f"{ADAPTER_DIR} (written only when v5 training fully completes)"
    )
    return out


# --------------------------------------------------------------------------- #
# The detached controller — wait -> gen -> council -> report -> showcase -> push
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=TIMEOUT_S, cpu=4.0, memory=8192,
              volumes={VOL_MOUNT: lora_vol, RESULTS_MOUNT: results_vol},
              secrets=[tfy_secret, hf_secret])
def finish(max_wait_s: int = DEFAULT_MAX_WAIT_S, council_concurrency: int = 6,
           force: bool = False) -> Dict[str, Any]:
    """One Mac-independent pass: block until v5 trains, then eval + persist."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        force=True)
    log = logging.getLogger("finish_v5")

    sys.path.insert(0, ROOT_REMOTE)
    os.chdir(ROOT_REMOTE)
    os.environ.setdefault("BENCH_DIR", f"{ROOT_REMOTE}/data/benchmark_gap803")

    results_vol.reload()
    Path(f"{RESULTS_MOUNT}/v5").mkdir(parents=True, exist_ok=True)

    # -- idempotency: already-done + single-flight lock ---------------------- #
    if Path(DONE_PATH).exists() and not force:
        log.info("DONE marker present; nothing to do (use force=True to re-run).")
        return {"status": "already_done", "state": _read_state()}

    if Path(LOCK_PATH).exists() and not force:
        try:
            lock = json.loads(Path(LOCK_PATH).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            lock = {}
        age = time.time() - float(lock.get("epoch", 0))
        if age < LOCK_STALE_S:
            log.warning("another finish() holds the lock (age %.0fs < %ds); exiting to avoid a "
                        "double-run.", age, LOCK_STALE_S)
            return {"status": "already_running", "lock": lock}
        log.warning("stale lock (age %.0fs); taking over.", age)

    call_id = os.environ.get("MODAL_TASK_ID", "unknown")
    Path(LOCK_PATH).write_text(json.dumps({"epoch": time.time(), "ts": _now(), "task": call_id}),
                               encoding="utf-8")
    results_vol.commit()

    state = _read_state() or {}
    state.update({"status": "running", "task": call_id, "started_ts": state.get("started_ts") or _now(),
                  "workspace": "chess-instructor-4"})
    _write_state(state)

    try:
        # -- PHASE 1: wait for the FINAL v5 adapter (training-done signal) ---- #
        log.info("PHASE wait: polling %s (every %ss, up to %ss)", ADAPTER_FILE, POLL_INTERVAL_S, max_wait_s)
        state["phase"] = "wait"; _write_state(state)
        t0 = time.time()
        while True:
            lora_vol.reload()
            if Path(ADAPTER_FILE).exists():
                log.info("adapter present after %.0fs: %s", time.time() - t0, sorted(os.listdir(ADAPTER_DIR)))
                break
            if time.time() - t0 > max_wait_s:
                state["status"] = "timeout_waiting_adapter"; _write_state(state)
                log.error("adapter never appeared within %ss; giving up (training likely still "
                          "running or failed).", max_wait_s)
                return {"status": "timeout_waiting_adapter", "waited_s": round(time.time() - t0)}
            time.sleep(POLL_INTERVAL_S)
        state["adapter_ready_ts"] = _now(); _write_state(state)

        # -- PHASE 2: finish the val gens (REUSE the deployed eval GPU app) --- #
        log.info("PHASE gen: val-gen rows so far = %d/%d", _gen_count(), N_VAL_SCENARIOS)
        state["phase"] = "gen"; _write_state(state)
        if _gen_count() < N_VAL_SCENARIOS:
            eval_gen = modal.Function.from_name(EVAL_APP, EVAL_FN)
            log.info("invoking %s::%s (serves v5 32B on A100, finishes val gens, then releases GPU)…",
                     EVAL_APP, EVAL_FN)
            res = eval_gen.remote(limit=0)
            log.info("val-gen result: %s", json.dumps(res, default=str))
            lora_vol.reload()
        n_gen = _gen_count()
        state["val_gen_rows"] = n_gen; _write_state(state)
        if n_gen < N_VAL_SCENARIOS:
            log.error("val-gen incomplete (%d/%d) after serve; aborting before council.",
                      n_gen, N_VAL_SCENARIOS)
            state["status"] = "gen_incomplete"; _write_state(state)
            return {"status": "gen_incomplete", "val_gen_rows": n_gen}

        # Copy the volume's ours_v5 gen into the repo path honest_v5 reads.
        Path(OURS_V5_LOCAL).parent.mkdir(parents=True, exist_ok=True)
        Path(OURS_V5_LOCAL).write_text(Path(GEN_REMOTE).read_text(encoding="utf-8"), encoding="utf-8")
        log.info("staged %d ours_v5 rows -> %s", n_gen, OURS_V5_LOCAL)

        # -- PHASE 3: council + report + showcase (REUSE honest_v5) ---------- #
        from argparse import Namespace
        import scripts.honest_v5 as HV5
        import scripts.honest_v4 as H  # same module honest_v5 re-pointed

        ns = Namespace(concurrency=council_concurrency, min_interval=0.05, timeout=300.0,
                       max_retries=8, judge_max_tokens=1600)

        log.info("PHASE slice: add ours_v5 to the 10-model v5 field")
        state["phase"] = "slice"; _write_state(state)
        HV5.cmd_slice(ns)

        log.info("PHASE council: blinded frontier panel (GPT-5.5 + Claude + Gemini via TFY) 0-10")
        state["phase"] = "council"; _write_state(state)
        H.cmd_council(ns)  # -> data/benchmark_honest/council_v5.jsonl

        log.info("PHASE report: v5-vs-v4-vs-4B verdict + vs-frontier proof")
        state["phase"] = "report"; _write_state(state)
        HV5.cmd_report(ns)  # -> RESULTS_HONEST_EVAL_V5.md + report_v5.json

        log.info("PHASE showcase: rebuild showcase.json seeded with v5 (distinct-tier AND beats-frontier)")
        state["phase"] = "showcase"; _write_state(state)
        H.cmd_showcase(ns)  # -> web/public/showcase.json (+ stats)

        # -- PHASE 4: persist (durable results volume + Hugging Face) -------- #
        log.info("PHASE push: results volume + Hugging Face")
        state["phase"] = "push"; _write_state(state)
        summary = _persist(log)
        state.update({"status": "complete", "completed_ts": _now(), "summary": summary})
        _write_state(state)
        Path(DONE_PATH).write_text(_now(), encoding="utf-8")
        results_vol.commit()
        log.info("DONE. verdict=%s", json.dumps(summary.get("verdict"), default=str))
        return {"status": "complete", **summary}
    except Exception as exc:  # noqa: BLE001
        import traceback
        log.error("finish() failed: %s\n%s", exc, traceback.format_exc())
        state["status"] = "error"; state["error"] = f"{type(exc).__name__}: {exc}"
        _write_state(state)
        raise
    finally:
        try:
            Path(LOCK_PATH).unlink(missing_ok=True)
            results_vol.commit()
        except Exception:  # noqa: BLE001
            pass


def _persist(log) -> Dict[str, Any]:
    """Copy the verdict bundle + reseeded showcase + adapter to the results
    volume, then push the bundle (HF dataset) + adapter (HF model)."""
    import shutil

    # ---- assemble the bundle on the durable results volume ---------------- #
    bundle = Path(BUNDLE_DIR)
    if bundle.exists():
        shutil.rmtree(bundle, ignore_errors=True)
    bundle.mkdir(parents=True, exist_ok=True)

    report = {}
    if Path(REPORT_JSON).exists():
        report = json.loads(Path(REPORT_JSON).read_text(encoding="utf-8"))
    verdict = report.get("verdict")

    # A compact, top-level verdict.json for quick reading.
    verdict_doc = {
        "generated_ts": _now(),
        "verdict": verdict,
        "v5_vs_v4_vs_4b_axes": report.get("v5_vs_v4_vs_4b_axes"),
        "distance_to_frontier": report.get("distance_to_frontier"),
        "vs_frontier_proof": report.get("vs_frontier_proof"),
        "council": report.get("council"),
        "field": report.get("field"),
    }
    (bundle / "verdict.json").write_text(json.dumps(verdict_doc, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    copied = []
    for src, name in [
        (MD_OUT, "RESULTS_HONEST_EVAL_V5.md"),
        (REPORT_JSON, "report_v5.json"),
        (COUNCIL_JSONL, "council_v5.jsonl"),
        (OURS_V5_LOCAL, "ours_v5.jsonl"),
        (SHOWCASE_JSON, "showcase.json"),
        (SHOWCASE_STATS, "showcase_v5_stats.json"),
    ]:
        if Path(src).exists():
            shutil.copyfile(src, bundle / name)
            copied.append(name)
    (bundle / "README.md").write_text(_bundle_readme(verdict_doc, copied), encoding="utf-8")
    log.info("bundle assembled (%s): %s", BUNDLE_DIR, copied)

    # A consolidated durable adapter copy on the results volume too.
    if Path(ADAPTER_DIR).exists():
        if Path(RESULTS_ADAPTER).exists():
            shutil.rmtree(RESULTS_ADAPTER, ignore_errors=True)
        shutil.copytree(ADAPTER_DIR, RESULTS_ADAPTER)
        log.info("adapter copied -> %s", RESULTS_ADAPTER)
    results_vol.commit()

    # ---- push to Hugging Face (durable, Mac-independent) ------------------ #
    hf: Dict[str, Any] = {"eval_repo": None, "adapter_repo": None}
    try:
        from huggingface_hub import HfApi
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=tok)

        api.create_repo(HF_EVAL_REPO, repo_type="dataset", exist_ok=True, private=True)
        api.upload_folder(folder_path=str(bundle), repo_id=HF_EVAL_REPO, repo_type="dataset",
                          commit_message="v5 (32B) Stage-2 verdict + reseeded showcase (cloud finish)")
        hf["eval_repo"] = f"https://huggingface.co/datasets/{HF_EVAL_REPO}"
        log.info("pushed eval bundle -> %s", hf["eval_repo"])

        if Path(ADAPTER_DIR).exists():
            api.create_repo(HF_ADAPTER_REPO, repo_type="model", exist_ok=True, private=True)
            api.upload_folder(folder_path=ADAPTER_DIR, repo_id=HF_ADAPTER_REPO, repo_type="model",
                              commit_message="v5 (32B) QLoRA adapter (cloud finish)")
            hf["adapter_repo"] = f"https://huggingface.co/{HF_ADAPTER_REPO}"
            log.info("pushed adapter -> %s", hf["adapter_repo"])
    except Exception as exc:  # noqa: BLE001
        log.error("HF push failed (results still on the results volume): %s", exc)
        hf["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "verdict": verdict,
        "distance_to_frontier": report.get("distance_to_frontier"),
        "vs_frontier_proof": report.get("vs_frontier_proof"),
        "hf": hf,
        "results_volume": {"volume": RESULTS_VOLUME, "bundle": BUNDLE_DIR, "adapter": RESULTS_ADAPTER},
        "bundle_files": copied + ["verdict.json", "README.md"],
    }


def _bundle_readme(verdict_doc: Dict[str, Any], files) -> str:
    v = verdict_doc.get("verdict") or {}
    d = verdict_doc.get("distance_to_frontier") or {}
    p = verdict_doc.get("vs_frontier_proof") or {}
    lines = [
        "# chess-coach v5 (Qwen3-32B QLoRA) — Stage-2 eval (cloud finish)",
        "",
        "Produced entirely on Modal (Mac-independent) by "
        "`src/orchestrate/finish_v5_modal.py`.",
        "",
        "## Verdict",
        f"- best_overall: **{v.get('best_overall')}**",
        f"- kept_moat_vs_v4: {v.get('kept_moat_vs_v4')}",
        f"- fixed_instructiveness_vs_v4: {v.get('fixed_instructiveness_vs_v4')}",
        f"- fixed_faithfulness_vs_v4: {v.get('fixed_faithfulness_vs_v4')}",
        "",
        "## vs-frontier + distinct-tier proof (the moat)",
        f"- distinct positions: {p.get('n_distinct')}; distinct AND diverge: "
        f"{p.get('n_distinct_and_diverge')}",
        f"- record vs best frontier: {p.get('wins')}W / {p.get('losses')}L / {p.get('ties')}T",
        f"- distance to frontier (instr rank gap): {d.get('gap')}",
        "",
        "## Files",
        *[f"- `{f}`" for f in (files + ["verdict.json"])],
        "",
        "## Reseed the live platform (when the Mac is back)",
        "```bash",
        "P=~/.venvs/mlx/bin/python",
        "# 1) pull this bundle",
        f"$P -m huggingface_hub download {HF_EVAL_REPO} --repo-type dataset "
        "--local-dir /tmp/v5_eval",
        "# 2) reseed the showcase the platform serves",
        "cp /tmp/v5_eval/showcase.json web/public/showcase.json",
        "# 3) restart the platform",
        "./run_platform.sh",
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# read_state (remote) — for the status entrypoint
# --------------------------------------------------------------------------- #
@app.function(image=cpu_image, timeout=120, volumes={RESULTS_MOUNT: results_vol})
def read_state() -> Dict[str, Any]:
    results_vol.reload()
    st = _read_state()
    st["_done_marker"] = Path(DONE_PATH).exists()
    st["_lock_present"] = Path(LOCK_PATH).exists()
    return st


# --------------------------------------------------------------------------- #
# Local entrypoints (trigger + inspect; everything else runs detached on Modal)
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def check() -> None:
    """Dry-run the trigger logic (no wait, no side effects)."""
    fn = modal.Function.from_name(APP_NAME, "preflight")
    print(json.dumps(fn.remote(), indent=2, default=str))


@app.local_entrypoint()
def arm(max_wait_h: float = 8.0, council_concurrency: int = 6, force: bool = False) -> None:
    """Spawn the DETACHED controller on the deployed app (survives a Mac shutdown)."""
    fn = modal.Function.from_name(APP_NAME, "finish")
    call = fn.spawn(max_wait_s=int(max_wait_h * 3600), council_concurrency=council_concurrency,
                    force=force)
    print(f"SPAWNED finish() call_id={call.object_id} on {APP_NAME} (detached on Modal). "
          f"It waits for {ADAPTER_FILE} on {LORA_VOLUME}, then evals + pushes results to "
          f"HF ({HF_EVAL_REPO} / {HF_ADAPTER_REPO}) + the {RESULTS_VOLUME} volume. "
          f"Safe to close the Mac. Inspect: modal run {Path(__file__).name}::status")


@app.local_entrypoint()
def status() -> None:
    """Print the durable pipeline state from the results volume."""
    fn = modal.Function.from_name(APP_NAME, "read_state")
    print(json.dumps(fn.remote(), indent=2, default=str))
