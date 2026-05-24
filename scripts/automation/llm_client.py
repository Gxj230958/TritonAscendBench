"""OpenAI-compatible LLM client with retry + budget tracking (Unit 1.5).

Thin HTTP wrapper around POST <endpoint>/v1/chat/completions (OpenAI
default) and POST <endpoint>/v1/messages (Anthropic fallback).

Public API
----------
* ``LLMClient(endpoint, api_style="openai", ...)``
* ``client.chat(messages, temperature=..., max_tokens=..., tools=None)``
    -> ``ChatResponse``
* ``ChatResponse(content, model, usage, stop_reason)`` dataclass
* ``LLMEmptyResponseError`` / ``LLMRetryExhaustedError`` /
  ``BudgetExceededError``

Retry policy (hand-rolled, no tenacity dep):
  retry on requests.Timeout / requests.ConnectionError / status in
  {429, 500, 502, 503, 504}; exponential backoff (2, 4, 8 -- capped 30s);
  3 attempts total. Fail-loud on 400 / 401 / 403 / 404.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger("automation.llm_client")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Base class for LLM client errors."""


class LLMEmptyResponseError(LLMError):
    """Response body parsed OK but ``content`` is null/empty."""


class LLMRetryExhaustedError(LLMError):
    """Retry budget exhausted; carries last exception or status."""


class BudgetExceededError(LLMError):
    """Cumulative usage would exceed ``max_budget_usd``."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    content: str
    model: str
    usage: dict[str, int]
    stop_reason: str = ""


@dataclass
class BudgetTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0
    # Optional per-token pricing (caller passes when known). Default 0 ->
    # budget is "soft" (counts tokens but never raises on cost).
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    max_budget_usd: float | None = None

    def add(self, usage: dict[str, int]) -> None:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_calls += 1
        self.estimated_cost_usd += (
            pt / 1000.0 * self.price_per_1k_prompt
            + ct / 1000.0 * self.price_per_1k_completion
        )

    def would_exceed(self, projected_prompt_tokens: int = 0) -> bool:
        if self.max_budget_usd is None:
            return False
        projected_cost = (
            self.estimated_cost_usd
            + projected_prompt_tokens / 1000.0 * self.price_per_1k_prompt
        )
        return projected_cost > self.max_budget_usd


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _safe_dict(d: Any) -> dict:
    return d if isinstance(d, dict) else {}


class LLMClient:
    """OpenAI-compatible chat client with retry + budget tracking."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_style: str = "openai",
        api_key: str | None = None,
        model: str = "default",
        timeout_s: int = 60,
        max_retries: int = 3,
        backoff_initial_s: float = 2.0,
        backoff_max_s: float = 30.0,
        budget: BudgetTracker | None = None,
        request_builder: Callable[..., dict] | None = None,
        response_parser: Callable[[dict], ChatResponse] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if api_style not in ("openai", "anthropic", "custom"):
            raise ValueError(f"unknown api_style: {api_style!r}")
        self.endpoint = endpoint.rstrip("/")
        self.api_style = api_style
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.budget = budget or BudgetTracker()
        self._request_builder = request_builder
        self._response_parser = response_parser
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public chat()
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: list | None = None,
    ) -> ChatResponse:
        # Cheap pre-check: rough token estimate (4 chars ~= 1 token) for
        # budget-pre-flight; over-estimate to err on the safe side.
        projected = sum(len(m.get("content", "")) for m in messages) // 3
        if self.budget.would_exceed(projected_prompt_tokens=projected):
            raise BudgetExceededError(
                f"projected prompt tokens={projected} would exceed budget"
            )

        if self.api_style == "openai":
            url = f"{self.endpoint}/v1/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools is not None:
                payload["tools"] = tools
        elif self.api_style == "anthropic":
            url = f"{self.endpoint}/v1/messages"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:  # custom
            if self._request_builder is None:
                raise ValueError("custom api_style requires request_builder")
            url, payload = self._request_builder(
                self.endpoint, self.model, messages,
                temperature=temperature, max_tokens=max_tokens, tools=tools,
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if self.api_style == "anthropic":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        data = self._post_with_retry(url, headers, payload)
        return self._parse_response(data)

    # ------------------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------------------
    def _post_with_retry(self, url: str, headers: dict, payload: dict) -> dict:
        last_exc: Exception | None = None
        backoff = self.backoff_initial_s
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, json=payload, headers=headers, timeout=self.timeout_s,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("LLM transient error (attempt %d/%d): %s",
                               attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(min(backoff, self.backoff_max_s))
                    backoff *= 2
                continue
            status = resp.status_code
            if 200 <= status < 300:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise LLMError(f"LLM response not JSON: {exc}") from exc
            if status in _RETRY_STATUSES:
                last_exc = LLMError(f"HTTP {status}: {resp.text[:200]}")
                logger.warning("LLM retryable status (attempt %d/%d): %s",
                               attempt, self.max_retries, last_exc)
                if attempt < self.max_retries:
                    time.sleep(min(backoff, self.backoff_max_s))
                    backoff *= 2
                continue
            # Non-retryable status -- fail loud.
            raise LLMError(f"HTTP {status} non-retryable: {resp.text[:200]}")
        raise LLMRetryExhaustedError(
            f"retries exhausted after {self.max_retries} attempts; last={last_exc}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, data: dict) -> ChatResponse:
        if self.api_style == "openai":
            choices = data.get("choices") or []
            if not choices:
                raise LLMEmptyResponseError("openai response missing choices")
            message = _safe_dict(choices[0].get("message"))
            content = message.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                raise LLMEmptyResponseError("openai response content is null/empty")
            usage = _safe_dict(data.get("usage"))
            self.budget.add(usage)
            return ChatResponse(
                content=str(content),
                model=str(data.get("model", self.model)),
                usage={
                    "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                },
                stop_reason=str(choices[0].get("finish_reason", "")),
            )
        elif self.api_style == "anthropic":
            blocks = data.get("content") or []
            if not blocks:
                raise LLMEmptyResponseError("anthropic response missing content")
            text = ""
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    text += str(b.get("text", ""))
            if not text.strip():
                raise LLMEmptyResponseError("anthropic response text is empty")
            usage = _safe_dict(data.get("usage"))
            # Anthropic field names differ
            mapped_usage = {
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
            }
            self.budget.add(mapped_usage)
            return ChatResponse(
                content=text,
                model=str(data.get("model", self.model)),
                usage=mapped_usage,
                stop_reason=str(data.get("stop_reason", "")),
            )
        else:  # custom
            if self._response_parser is None:
                raise ValueError("custom api_style requires response_parser")
            return self._response_parser(data)


__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "ChatResponse",
    "LLMClient",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMRetryExhaustedError",
]
