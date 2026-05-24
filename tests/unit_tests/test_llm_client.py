"""Tests for ``automation.llm_client`` (Unit 1.5).

Uses ``unittest.mock.patch`` on ``requests.Session.post`` to stub HTTP.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"


@pytest.fixture(autouse=True)
def _ensure_scripts_on_path():
    inserted = False
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
        inserted = True
    yield
    if inserted:
        try:
            sys.path.remove(str(_SCRIPTS))
        except ValueError:
            pass


def _make_resp(status: int, payload):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = repr(payload)
    return r


def test_happy_openai_chat():
    from automation import llm_client

    client = llm_client.LLMClient(
        "http://llm.local", api_style="openai", model="qwen-2.5-coder",
        backoff_initial_s=0.0,
    )
    payload = {
        "model": "qwen-2.5-coder",
        "choices": [{"message": {"content": "hello world"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }
    with patch.object(client._session, "post", return_value=_make_resp(200, payload)):
        resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "hello world"
    assert resp.usage["prompt_tokens"] == 10
    assert client.budget.prompt_tokens == 10
    assert client.budget.completion_tokens == 20


def test_happy_anthropic_chat():
    from automation import llm_client

    client = llm_client.LLMClient(
        "http://claude.local", api_style="anthropic", model="claude-3-5-sonnet",
        backoff_initial_s=0.0,
    )
    payload = {
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": "anthro-output"}],
        "usage": {"input_tokens": 7, "output_tokens": 11},
        "stop_reason": "end_turn",
    }
    with patch.object(client._session, "post", return_value=_make_resp(200, payload)):
        resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "anthro-output"
    assert resp.usage["prompt_tokens"] == 7
    assert resp.usage["completion_tokens"] == 11


def test_empty_content_raises():
    from automation import llm_client

    client = llm_client.LLMClient("http://x", api_style="openai", backoff_initial_s=0.0)
    payload = {
        "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
    }
    with patch.object(client._session, "post", return_value=_make_resp(200, payload)):
        with pytest.raises(llm_client.LLMEmptyResponseError):
            client.chat([{"role": "user", "content": "hi"}])


def test_usage_missing_still_works():
    from automation import llm_client

    client = llm_client.LLMClient("http://x", api_style="openai", backoff_initial_s=0.0)
    payload = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        # no "usage" key
    }
    with patch.object(client._session, "post", return_value=_make_resp(200, payload)):
        resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert resp.usage["prompt_tokens"] == 0


def test_503_retries_then_exhausts():
    from automation import llm_client

    client = llm_client.LLMClient(
        "http://x", api_style="openai", backoff_initial_s=0.0, max_retries=3,
    )
    resp_503 = _make_resp(503, {"error": "service unavailable"})
    with patch.object(client._session, "post", return_value=resp_503) as post:
        with pytest.raises(llm_client.LLMRetryExhaustedError):
            client.chat([{"role": "user", "content": "hi"}])
    assert post.call_count == 3


def test_400_does_not_retry():
    from automation import llm_client

    client = llm_client.LLMClient(
        "http://x", api_style="openai", backoff_initial_s=0.0, max_retries=3,
    )
    resp_400 = _make_resp(400, {"error": "bad request"})
    with patch.object(client._session, "post", return_value=resp_400) as post:
        with pytest.raises(llm_client.LLMError, match="non-retryable"):
            client.chat([{"role": "user", "content": "hi"}])
    assert post.call_count == 1


def test_timeout_then_success():
    import requests
    from automation import llm_client

    client = llm_client.LLMClient(
        "http://x", api_style="openai", backoff_initial_s=0.0, max_retries=3,
    )
    good = _make_resp(
        200, {"choices": [{"message": {"content": "after-retry"}, "finish_reason": "stop"}]}
    )
    sequence = [requests.Timeout("slow"), good]
    with patch.object(client._session, "post", side_effect=sequence):
        resp = client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "after-retry"


def test_budget_exceeded_pre_flight_raises():
    from automation import llm_client

    bt = llm_client.BudgetTracker(
        price_per_1k_prompt=0.001,
        price_per_1k_completion=0.002,
        max_budget_usd=0.0001,  # tiny budget
    )
    client = llm_client.LLMClient(
        "http://x", api_style="openai", budget=bt, backoff_initial_s=0.0,
    )
    # No mock needed -- should raise BudgetExceededError before requesting.
    big_message = "a" * 10000
    with pytest.raises(llm_client.BudgetExceededError):
        client.chat([{"role": "user", "content": big_message}])


def test_budget_accumulates_across_calls():
    from automation import llm_client

    client = llm_client.LLMClient("http://x", api_style="openai", backoff_initial_s=0.0)
    payload = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    with patch.object(client._session, "post", return_value=_make_resp(200, payload)):
        client.chat([{"role": "user", "content": "hi"}])
        client.chat([{"role": "user", "content": "hi"}])
    assert client.budget.prompt_tokens == 10
    assert client.budget.completion_tokens == 6
    assert client.budget.total_calls == 2


def test_invalid_api_style_raises():
    from automation import llm_client

    with pytest.raises(ValueError):
        llm_client.LLMClient("http://x", api_style="banana")
