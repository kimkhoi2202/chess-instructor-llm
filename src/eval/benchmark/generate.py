"""Generation phase: 5 models x 2 conditions x N scenarios, resumable + costed.

Local models run sequentially (one MLX load each, then all their items). Frontier
models run concurrently through a shared rate limiter. Every completed generation
is appended to ``generations.jsonl`` immediately and keyed by
``(scenario_id, model, condition)``, so a crash or a killed API call never loses
or double-charges completed work — a rerun just fills the gaps.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from . import config as bcfg
from .backends import MLXLocal, RateLimiter, TFYChat, make_tfy_client
from .io_utils import append_jsonl, done_keys
from .prompts import build_user_prompt, load_system_prompt

log = logging.getLogger("benchmark.generate")

Unit = Tuple[Dict[str, Any], str, str]  # (scenario, model_key, condition)


def _pending(
    scenarios: Sequence[Dict[str, Any]],
    model_keys: Sequence[str],
    conditions: Sequence[str],
) -> List[Unit]:
    done = done_keys(bcfg.GENERATIONS_PATH, ["scenario_id", "model", "condition"])
    units: List[Unit] = []
    for scn in scenarios:
        for mk in model_keys:
            for cond in conditions:
                if (scn["id"], mk, cond) not in done:
                    units.append((scn, mk, cond))
    return units


def _persist(scn: Dict[str, Any], model_key: str, condition: str, text: str,
             usage: Dict[str, int]) -> None:
    append_jsonl(
        bcfg.GENERATIONS_PATH,
        {
            "scenario_id": scn["id"],
            "model": model_key,
            "condition": condition,
            "tier": scn["tier"],
            "phase": scn["phase"],
            "severity": scn["severity"],
            "output": text,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    )


def _run_local(system: str, model_key: str, units: List[Unit]) -> Tuple[int, int]:
    """Generate all local units for one MLX model (loads it once)."""
    mine = [u for u in units if u[1] == model_key]
    if not mine:
        return 0, 0
    model = bcfg.MODELS[model_key]
    log.info("loading local model %s (%s) for %d items ...", model_key, model.ident, len(mine))
    backend = MLXLocal(model.ident, max_tokens=bcfg.GEN_MAX_TOKENS_LOCAL)
    ok = fail = 0
    for i, (scn, _mk, cond) in enumerate(mine, 1):
        t0 = time.time()
        try:
            user = build_user_prompt(scn, cond)
            text, usage = backend.complete(system, user)
            _persist(scn, model_key, cond, text, usage)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - one item must not abort the model
            fail += 1
            log.error("local %s %s/%s failed: %s", model_key, scn["id"], cond, exc)
        if i % 20 == 0 or i == len(mine):
            log.info("  %s: %d/%d (%.1fs/item)", model_key, i, len(mine),
                     (time.time() - t0))
    return ok, fail


def _run_frontier(
    system: str,
    model_keys: Sequence[str],
    units: List[Unit],
    *,
    concurrency: int,
    min_interval: float,
    timeout: float,
    max_retries: int,
) -> Tuple[int, int]:
    """Generate all frontier units concurrently (shared rate limiter)."""
    mine = [u for u in units if u[1] in model_keys]
    if not mine:
        return 0, 0
    client = make_tfy_client(timeout)
    limiter = RateLimiter(min_interval)
    clients = {
        mk: TFYChat(client, model_id=bcfg.MODELS[mk].ident,
                    max_tokens=bcfg.GEN_MAX_TOKENS_TFY, max_retries=max_retries,
                    limiter=limiter, reasoning_effort=bcfg.MODELS[mk].reasoning_effort)
        for mk in model_keys
    }
    log.info("frontier generation: %d items, %d workers", len(mine), concurrency)

    ok = fail = done = 0

    def _task(unit: Unit) -> Tuple[Unit, str, Dict[str, int]]:
        scn, mk, cond = unit
        user = build_user_prompt(scn, cond)
        text, usage = clients[mk].complete(system, user)
        return unit, text, usage

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_task, u): u for u in mine}
        for fut in as_completed(futures):
            scn, mk, cond = futures[fut]
            done += 1
            try:
                _unit, text, usage = fut.result()
                _persist(scn, mk, cond, text, usage)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - skip; a rerun retries it
                fail += 1
                log.error("frontier %s %s/%s failed: %s", mk, scn["id"], cond, exc)
            if done % 25 == 0 or done == len(mine):
                log.info("  frontier: %d/%d (ok=%d fail=%d)", done, len(mine), ok, fail)
    return ok, fail


def run_generation(
    scenarios: Sequence[Dict[str, Any]],
    model_keys: Sequence[str],
    conditions: Sequence[str],
    *,
    concurrency: int = 6,
    min_interval: float = 0.05,
    timeout: float = 300.0,
    max_retries: int = 4,
) -> Dict[str, int]:
    """Fill every missing (scenario, model, condition) generation. Returns counts."""
    units = _pending(scenarios, model_keys, conditions)
    total_planned = len(scenarios) * len(model_keys) * len(conditions)
    log.info("generation: %d pending of %d total units", len(units), total_planned)
    if not units:
        return {"ok": 0, "fail": 0, "pending": 0}

    system = load_system_prompt()
    local_keys = [mk for mk in model_keys if bcfg.MODELS[mk].kind == "mlx"]
    tfy_keys = [mk for mk in model_keys if bcfg.MODELS[mk].kind == "tfy"]

    ok = fail = 0
    for mk in local_keys:
        o, f = _run_local(system, mk, units)
        ok += o
        fail += f
    if tfy_keys:
        o, f = _run_frontier(
            system, tfy_keys, units,
            concurrency=concurrency, min_interval=min_interval,
            timeout=timeout, max_retries=max_retries,
        )
        ok += o
        fail += f

    log.info("generation done: ok=%d fail=%d", ok, fail)
    return {"ok": ok, "fail": fail, "pending": len(units)}
