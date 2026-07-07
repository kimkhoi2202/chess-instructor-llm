"""Model backends: local MLX competitors + frontier gateway competitors/judges.

Two ways to turn (system, user) into text:

* :class:`MLXLocal` — wraps the project's existing ``MLXBackend`` (greedy, thinking
  disabled, ``<think>`` stripped). Free; used for OURS and BASE.
* :class:`TFYChat` — the TrueFoundry OpenAI-compatible ``chat.completions`` path
  used for the three frontier competitors and, identically, for the three council
  judges. It **never sends ``temperature``** (the gateway 400s on it for some
  models), retries transient errors with the same backoff policy as
  ``src/teacher/generate.py``, and records token usage for the cost readout.

Every call returns ``(text, usage)`` where ``usage`` is
``{"prompt_tokens": int, "completion_tokens": int}`` (zeros for local).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import openai
from dotenv import load_dotenv
from openai import OpenAI

from config import settings
from src.eval.evaluate import MLXBackend, _strip_think
from src.teacher.generate import RateLimiter, _RETRYABLE, _backoff

log = logging.getLogger("benchmark.backends")

Usage = Dict[str, int]


def _zero_usage() -> Usage:
    return {"prompt_tokens": 0, "completion_tokens": 0}


# --------------------------------------------------------------------------- #
# Local MLX
# --------------------------------------------------------------------------- #


class MLXLocal:
    """A local MLX competitor (loads once, greedy, deterministic)."""

    def __init__(self, model_path: str, *, max_tokens: int) -> None:
        self._backend = MLXBackend(model_path, max_tokens=max_tokens)

    def complete(self, system: str, user: str) -> Tuple[str, Usage]:
        text = self._backend.generate(system, user)
        return text.strip(), _zero_usage()


# --------------------------------------------------------------------------- #
# Frontier via the TrueFoundry gateway
# --------------------------------------------------------------------------- #


def make_tfy_client(timeout: float) -> OpenAI:
    """Build an OpenAI client pointed at the TrueFoundry gateway (key from .env)."""
    load_dotenv(settings.ROOT / ".env")
    key = os.environ.get("TFY_API_KEY")
    base = os.environ.get("TFY_BASE_URL")
    if not key or not base:
        raise RuntimeError("TFY_API_KEY / TFY_BASE_URL missing from ROOT/.env")
    # We own retries/backoff, so disable the SDK's built-in ones.
    return OpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=0)


def _extract_usage(resp: Any) -> Usage:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return _zero_usage()
    prompt = getattr(usage, "prompt_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", None)
    if completion is None:
        completion = getattr(usage, "output_tokens", 0) or 0
    return {"prompt_tokens": int(prompt), "completion_tokens": int(completion)}


class TFYChat:
    """Frontier chat.completions client (no temperature; retries; usage tracked).

    ``reasoning_effort`` is passed through when set (required for GPT-5.5, which
    otherwise spends its whole token budget reasoning and returns empty content).
    If the gateway rejects the param for a given model it is dropped for the rest
    of the run and the call retried without it.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model_id: str,
        max_tokens: int,
        max_retries: int,
        limiter: RateLimiter,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self._limiter = limiter
        self._reasoning_effort = reasoning_effort

    def _create(self, system: str, user: str):
        kwargs: Dict[str, Any] = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
        )
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            return self._client.chat.completions.create(**kwargs)
        except (openai.BadRequestError, TypeError) as exc:
            if self._reasoning_effort:
                log.warning("%s rejected reasoning_effort (%s); dropping it",
                            self.model_id, type(exc).__name__)
                self._reasoning_effort = None
                kwargs.pop("reasoning_effort", None)
                return self._client.chat.completions.create(**kwargs)
            raise

    def complete(self, system: str, user: str) -> Tuple[str, Usage]:
        """Return ``(text, usage)`` from the model (retrying transient failures)."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._create(system, user)
                content = _strip_think(resp.choices[0].message.content or "").strip()
                usage = _extract_usage(resp)
                if not content:
                    raise ValueError("empty model response")
                return content, usage
            except _RETRYABLE as exc:
                last_exc = exc
                delay = _backoff(attempt)
                log.warning(
                    "%s transient (%s) attempt %d/%d; retry in %.1fs",
                    self.model_id, type(exc).__name__, attempt + 1,
                    self.max_retries + 1, delay,
                )
                time.sleep(delay)
            except ValueError as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = _backoff(attempt)
                log.warning(
                    "%s empty/parse (%s) attempt %d/%d; retry in %.1fs",
                    self.model_id, exc, attempt + 1, self.max_retries + 1, delay,
                )
                time.sleep(delay)
        raise last_exc if last_exc is not None else RuntimeError("tfy call failed")
