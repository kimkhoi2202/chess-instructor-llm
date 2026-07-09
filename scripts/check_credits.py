"""Cheap, repeatable credit monitor for the autonomous training loop.

For each monitored Modal workspace it reports:
  * spend this billing range (summed from `modal billing report --json`),
  * estimated headroom = starting budget - spend,
  * whether the training app is still RUNNING (`modal app list --json`),
and emits one machine-readable line per workspace:

    VERDICT <workspace> OK|LOW|PORT

  OK   = headroom >= LOW_THRESHOLD (app-state informational: a stopped app
         with healthy headroom just means the training FINISHED).
  LOW  = headroom < LOW_THRESHOLD (warn; keep going, but a port is near).
  PORT = headroom <= PORT_THRESHOLD (credits exhausted). Port ONLY on PORT;
         a finished/idle app with good headroom is NOT a port case.
  ERROR (extra, non-actionable) = a modal call failed; the loop should do
         nothing and retry next tick rather than treat a transient CLI/network
         failure as a real PORT.

FOOTGUN handled internally: `.env`'s bare MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
belong to kim-lam and OVERRIDE MODAL_PROFILE. Every modal subprocess is run
with those two vars stripped from its environment and MODAL_PROFILE set to the
target workspace. Token values are never read or printed.

Exit code: 0 if every workspace is OK, 1 otherwise (any LOW / PORT / ERROR).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# --- Known budgets & training apps (easy to update) -------------------------
# budget = total credits the workspace started with (USD).
# app_id = the training app whose liveness gates the PORT decision.
WORKSPACES: dict[str, dict[str, object]] = {
    "chess-instructor-3": {
        # Modal capped new workspaces: this is the LAST one, ~$30 grant like the
        # others (NOT $300 — the 10-workspace plan is dead). Conservative budget
        # so the monitor warns EARLY rather than stranding it.
        "budget": 30.0,
        "app_id": "ap-mylaYjsYAJOGmqO0OPNu4w",  # v4 (32B) eval gen — finishes the honest VAL slice
        "label": "32B eval + demo (primary, ~$30, LAST workspace)",
    },
    "chess-instructor-4": {
        # Newest workspace (~$30). chess-instructor-5 blocked by Modal's
        # membership cap, so this is the ceiling: total usable ~$87.
        "budget": 30.0,
        "app_id": "ap-mduWYpJIB1cYGT3x6rwffz",  # v5 32B QLoRA training (chess-coach-qlora-v5, live run)
        "label": "32B headroom (~$30)",
    },
    "chess-instructor-2": {
        "budget": 30.0,
        "app_id": "ap-oOcYFot6Db8GcBvfsk0Mcn",
        "label": "4B QLoRA training",
    },
    "chess-instructor": {
        "budget": 30.0,
        "app_id": "ap-pFAFVeAwSlbCZJyyPnVdmU",
        "label": "v4 32B QLoRA training",
    },
}

# Spend window. "this month" matches how the workspaces were funded (all spend
# so far is within the current calendar month). Change if a run spans a month
# boundary (a new month would reset spend to ~0 and overstate headroom).
BILLING_RANGE = "this month"

LOW_THRESHOLD = 5.0   # headroom below this -> LOW
PORT_THRESHOLD = 1.0  # headroom at/below this (near-zero) -> PORT

# States that mean the training app is no longer doing work.
NON_RUNNING_MARKERS = ("stopped", "crash", "error", "terminat", "timeout", "fail")

# kim-lam is billing-blocked; guard against ever selecting it.
FORBIDDEN_PROFILE = "kim-lam"

SUBPROCESS_TIMEOUT_S = 90


def _modal_bin() -> str:
    """Resolve the modal CLI: sibling of the running interpreter, then PATH,
    then the known venv location."""
    sibling = os.path.join(os.path.dirname(sys.executable), "modal")
    if os.path.exists(sibling):
        return sibling
    from shutil import which

    return which("modal") or "/Users/khoilam/.venvs/mlx/bin/modal"


def _clean_env(profile: str) -> dict[str, str]:
    """Env for a modal call: strip the bare kim-lam tokens (the footgun) and
    pin MODAL_PROFILE to the target workspace."""
    if profile == FORBIDDEN_PROFILE:
        raise ValueError(f"refusing to run against forbidden profile {profile!r}")
    env = dict(os.environ)
    env.pop("MODAL_TOKEN_ID", None)
    env.pop("MODAL_TOKEN_SECRET", None)
    env["MODAL_PROFILE"] = profile
    return env


def _run_json(args: list[str], profile: str):
    """Run `modal <args> --json` for a profile and return parsed JSON.
    Raises RuntimeError with a short message on any failure."""
    cmd = [_modal_bin(), *args, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            env=_clean_env(profile),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"timeout after {SUBPROCESS_TIMEOUT_S}s")
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(msg[-1] if msg else f"exit {proc.returncode}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("non-JSON output from modal")


def get_spend(profile: str) -> float:
    """Total spend (USD) over BILLING_RANGE = sum of all line-item costs."""
    rows = _run_json(["billing", "report", "--for", BILLING_RANGE], profile)
    return sum(float(r.get("cost", 0) or 0) for r in rows)


def get_app_status(profile: str, app_id: str) -> tuple[bool, str]:
    """Return (is_running, state_str) for the training app."""
    apps = _run_json(["app", "list"], profile)
    for a in apps:
        if a.get("app_id") == app_id:
            state = str(a.get("state", "")).strip()
            running = not any(m in state.lower() for m in NON_RUNNING_MARKERS)
            return running, state or "unknown"
    return False, "not found"


def verdict_for(headroom: float, running: bool) -> str:
    # PORT is a CREDIT decision ONLY: trigger when headroom is near-zero
    # (credits exhausted). A stopped app with healthy headroom means the
    # training simply FINISHED (or a non-credit crash) — that is NOT a
    # workspace-port case, so `running` is informational and never forces PORT.
    # (A genuine credit-kill also drives headroom to ~0, so it's still caught.)
    if headroom <= PORT_THRESHOLD:
        return "PORT"
    if headroom < LOW_THRESHOLD:
        return "LOW"
    return "OK"


def check_workspace(name: str, cfg: dict[str, object]) -> str:
    budget = float(cfg["budget"])  # type: ignore[arg-type]
    app_id = str(cfg["app_id"])
    label = str(cfg.get("label", ""))
    header = f"== {name} ({label}) =="
    try:
        spend = get_spend(name)
        running, state = get_app_status(name, app_id)
    except RuntimeError as e:
        print(header)
        print(f"  ERROR: {e}")
        print(f"VERDICT {name} ERROR")
        return "ERROR"

    headroom = budget - spend
    v = verdict_for(headroom, running)
    run_txt = "RUNNING" if running else "STOPPED/ERRORED"
    print(header)
    print(f"  spend ({BILLING_RANGE}): ${spend:,.2f}")
    print(f"  budget:              ${budget:,.2f}")
    print(f"  headroom (est):      ${headroom:,.2f}")
    print(f"  training app:        {app_id}  {run_txt} ({state})")
    print(f"VERDICT {name} {v}")
    return v


def main() -> int:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"# credit check {ts}  (LOW<${LOW_THRESHOLD:g}, PORT<=${PORT_THRESHOLD:g} headroom; app-state informational)")
    verdicts = []
    for name, cfg in WORKSPACES.items():
        print()
        verdicts.append(check_workspace(name, cfg))
    return 0 if all(v == "OK" for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
