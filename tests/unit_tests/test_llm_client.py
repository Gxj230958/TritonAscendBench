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


# ---------------------------------------------------------------------------
# SSHCLIClient
# ---------------------------------------------------------------------------

def _make_completed(stdout: str, stderr: str = "", returncode: int = 0):
    """Helper: synthesize a subprocess.CompletedProcess-like object."""
    cp = MagicMock()
    cp.stdout = stdout
    cp.stderr = stderr
    cp.returncode = returncode
    return cp


def test_ssh_cli_client_requires_prompt_placeholder():
    """cli_cmd_template missing '{prompt}' -> ValueError fail-loud."""
    from automation import llm_client

    # Example uses a generic placeholder CLI name; the bench never
    # hardcodes a specific CLI binary.
    with pytest.raises(ValueError, match="prompt"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="some-cli exec",  # missing {prompt}
        )


# ---------------------------------------------------------------------------
# SSHCLIClient template validation: literal-replace, brace-friendly,
# placeholder-count, quote-wrapping (MUST-FIX 1 + MUST-FIX 2 regression).
# ---------------------------------------------------------------------------

def test_template_with_shell_vars_accepted_at_init():
    """Template containing shell variable expansion stays raw: no KeyError.

    str.format would raise KeyError('HOME') on '${HOME}/cli {prompt}'.
    The client must use literal .replace so the variable survives to
    the remote shell unchanged.
    """
    from automation import llm_client

    template = "${HOME}/cli exec {prompt}"
    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template=template,
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _make_completed(stdout="reply", returncode=0)

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        client.chat([{"role": "user", "content": "hi"}])

    rendered = captured["argv"][-1]
    # Literal '${HOME}' must be preserved verbatim for remote shell.
    assert "${HOME}/cli exec" in rendered


def test_template_with_json_brace_accepted():
    """Template containing JSON literal '{"k":"v"}' must not raise.

    str.format would barf on the JSON braces. Literal replace passes
    them through to the remote CLI as-is.
    """
    from automation import llm_client

    template = 'some-cli --opts {"k":"v"} exec {prompt}'
    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template=template,
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _make_completed(stdout="reply", returncode=0)

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        client.chat([{"role": "user", "content": "hi"}])

    rendered = captured["argv"][-1]
    assert '{"k":"v"}' in rendered


def test_template_with_zero_placeholders_rejected():
    """Zero '{prompt}' placeholders -> __init__ raises ValueError."""
    from automation import llm_client

    with pytest.raises(ValueError, match="exactly once"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="some-cli exec",
        )


def test_template_with_two_placeholders_rejected():
    """Two '{prompt}' placeholders -> __init__ raises ValueError."""
    from automation import llm_client

    with pytest.raises(ValueError, match="exactly once"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="some-cli {prompt} {prompt}",
        )


def test_template_with_double_quoted_prompt_rejected():
    """'{prompt}' wrapped in double quotes -> __init__ rejects."""
    from automation import llm_client

    with pytest.raises(ValueError, match="do NOT wrap"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template='some-cli exec "{prompt}"',
        )


def test_template_with_single_quoted_prompt_rejected():
    """'{prompt}' wrapped in single quotes -> __init__ rejects."""
    from automation import llm_client

    with pytest.raises(ValueError, match="do NOT wrap"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="some-cli exec '{prompt}'",
        )


def test_template_without_quotes_around_prompt_accepted():
    """Bare '{prompt}' (no surrounding quotes) is the happy path."""
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )
    assert client.cli_cmd_template == "some-cli exec {prompt}"


def test_template_with_adjacent_non_quote_chars_accepted():
    """Non-quote chars adjacent to '{prompt}' are legitimate.

    Edge case: 'X{prompt}Y' is unusual but valid -- X/Y are not
    shell quote chars, so the validation must let it through.
    """
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec X{prompt}Y",
    )
    assert "{prompt}" in client.cli_cmd_template


def test_template_outer_double_quotes_rejected():
    """Outer ``"..."`` wrapping the whole tail (incl. {prompt}) rejected.

    Regression for the "adjacent-char-only" gap: ``cli exec "prefix
    {prompt} suffix"`` has space adjacent to ``{prompt}``, so the
    narrow neighbour check let it through, but the placeholder is
    still inside an outer double-quoted segment -- the bench's shlex
    quoting would nest inside that quote and produce a literal quote
    on the wire.
    """
    from automation import llm_client

    with pytest.raises(ValueError, match="do NOT wrap"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template='cli exec "prefix {prompt} suffix"',
        )


def test_template_outer_single_quotes_rejected():
    """Outer ``'...'`` wrapping the whole tail (incl. {prompt}) rejected."""
    from automation import llm_client

    with pytest.raises(ValueError, match="do NOT wrap"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="cli exec 'prefix {prompt} suffix'",
        )


def test_template_nested_double_inside_single_accepted_for_literal():
    """Closed-then-reopened quotes around literals do not poison {prompt}.

    ``cli echo '$VAR' exec {prompt}`` opens + closes a single-quoted
    segment around ``$VAR`` (so the literal ``$`` survives without
    shell expansion), then leaves the placeholder fully outside any
    quoted segment. The quote-depth tracker must report "not currently
    quoted" at the placeholder position and let the template through.
    """
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="cli echo '$VAR' exec {prompt}",
    )
    assert "{prompt}" in client.cli_cmd_template


def test_template_unclosed_quote_rejected():
    """Unclosed quote before {prompt} is fatal misconfiguration.

    ``cli exec " {prompt}`` opens a double-quoted segment that never
    closes; the placeholder sits inside that segment, so the
    quote-depth check rejects it. Same code path as outer-quoted
    rejection -- error message contains ``do NOT wrap``.
    """
    from automation import llm_client

    with pytest.raises(ValueError, match="do NOT wrap"):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template='cli exec " {prompt}',
        )


def test_ssh_cli_client_requires_ssh_host():
    from automation import llm_client

    with pytest.raises(ValueError):
        llm_client.SSHCLIClient(
            ssh_host="",
            cli_cmd_template="some-cli exec {prompt}",
        )


def test_ssh_cli_client_requires_cli_cmd_template():
    from automation import llm_client

    with pytest.raises(ValueError):
        llm_client.SSHCLIClient(
            ssh_host="user@c-host",
            cli_cmd_template="",
        )


def test_ssh_cli_client_builds_correct_argv():
    """Argv must be ssh + options + host + substituted template."""
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _make_completed(stdout="reply", returncode=0)

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        client.chat([{"role": "user", "content": "hello"}])

    argv = captured["argv"]
    assert argv[0] == "ssh"
    # BatchMode=yes default option present
    assert "BatchMode=yes" in argv
    # ssh host argument
    assert "user@c-host" in argv
    # Final element is the rendered CLI command string (one ssh arg)
    last = argv[-1]
    assert last.startswith("some-cli exec ")
    assert "hello" in last


def test_ssh_cli_client_shlex_quotes_prompt():
    """Prompt with shell metacharacters must be safely shlex-quoted."""
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _make_completed(stdout="reply", returncode=0)

    nasty = 'evil"; rm -rf / #'
    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        client.chat([{"role": "user", "content": nasty}])

    rendered = captured["argv"][-1]
    # The dangerous literal must NOT appear as an unquoted shell token.
    # shlex.quote wraps the whole prompt in single quotes (and escapes
    # any embedded single quote). We assert the dangerous fragment is
    # inside a single-quoted region rather than free-floating.
    assert nasty in rendered  # the literal is preserved verbatim
    # The substring around the user content must be single-quoted.
    assert "'" in rendered
    # Sanity: the rendered ssh argument is one shell-quoted string;
    # shlex.split must roundtrip back to the original tokens.
    import shlex as _shlex
    tokens = _shlex.split(rendered)
    assert tokens[0] == "some-cli"
    assert tokens[1] == "exec"
    # The full prompt is one shell token (role tag + content).
    assert nasty in tokens[2]


def test_ssh_cli_client_timeout_raises_llm_timeout():
    """subprocess.TimeoutExpired -> LLMTimeoutError, not generic LLMError."""
    import subprocess as _sp

    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
        timeout_seconds=5,
    )

    def boom(argv, **kwargs):
        raise _sp.TimeoutExpired(cmd=argv, timeout=5)

    with patch.object(llm_client.subprocess, "run", side_effect=boom):
        with pytest.raises(llm_client.LLMTimeoutError):
            client.chat([{"role": "user", "content": "hi"}])


def test_ssh_cli_client_nonzero_rc_raises_transport_error():
    """rc != 0 -> LLMTransportError carrying stderr head for triage."""
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )

    def fake_run(argv, **kwargs):
        return _make_completed(
            stdout="", stderr="auth failed: no key", returncode=255,
        )

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        with pytest.raises(llm_client.LLMTransportError) as exc_info:
            client.chat([{"role": "user", "content": "hi"}])
    assert "rc=255" in str(exc_info.value)
    assert "auth failed" in str(exc_info.value)


def test_ssh_cli_client_empty_stdout_raises_empty_response():
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )

    def fake_run(argv, **kwargs):
        return _make_completed(stdout="   \n", returncode=0)

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        with pytest.raises(llm_client.LLMEmptyResponseError):
            client.chat([{"role": "user", "content": "hi"}])


def test_ssh_cli_client_records_budget():
    """Successful chat increments total_calls + wall_time_seconds."""
    from automation import llm_client

    client = llm_client.SSHCLIClient(
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )

    def fake_run(argv, **kwargs):
        return _make_completed(stdout="ok", returncode=0)

    with patch.object(llm_client.subprocess, "run", side_effect=fake_run):
        client.chat([{"role": "user", "content": "first"}])
        client.chat([{"role": "user", "content": "second"}])

    assert client.budget.total_calls == 2
    assert client.budget.wall_time_seconds >= 0.0
    # CLI does not expose tokens; price fields stay zero.
    assert client.budget.prompt_tokens == 0
    assert client.budget.completion_tokens == 0
    assert client.budget.estimated_cost_usd == 0.0


def test_ssh_cli_client_concat_messages_format():
    """_concat_messages renders [role]\\n<content> blocks, blank-line sep."""
    from automation import llm_client

    text = llm_client.SSHCLIClient._concat_messages([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "ping"},
    ])
    assert "[system]\nbe helpful" in text
    assert "[user]\nping" in text
    # Blocks separated by a blank line.
    assert "\n\n" in text


# ---------------------------------------------------------------------------
# Factory: create_llm_client
# ---------------------------------------------------------------------------

def test_factory_http_returns_llm_client():
    from automation import llm_client

    c = llm_client.create_llm_client(
        transport="http", endpoint="http://localhost:8080",
    )
    assert isinstance(c, llm_client.LLMClient)


def test_factory_ssh_cli_returns_ssh_cli_client():
    from automation import llm_client

    c = llm_client.create_llm_client(
        transport="ssh-cli",
        ssh_host="user@c-host",
        cli_cmd_template="some-cli exec {prompt}",
    )
    assert isinstance(c, llm_client.SSHCLIClient)


def test_factory_unknown_transport_raises():
    from automation import llm_client

    with pytest.raises(ValueError, match="unknown LLM transport"):
        llm_client.create_llm_client(transport="bogus")


def test_factory_http_missing_endpoint_raises():
    from automation import llm_client

    with pytest.raises(ValueError, match="endpoint"):
        llm_client.create_llm_client(transport="http", endpoint=None)


def test_factory_ssh_cli_missing_host_raises():
    from automation import llm_client

    with pytest.raises(ValueError, match="ssh_host"):
        llm_client.create_llm_client(
            transport="ssh-cli",
            ssh_host=None,
            cli_cmd_template="some-cli exec {prompt}",
        )


def test_factory_ssh_cli_missing_cli_cmd_raises():
    from automation import llm_client

    with pytest.raises(ValueError, match="cli_cmd_template"):
        llm_client.create_llm_client(
            transport="ssh-cli",
            ssh_host="user@c-host",
            cli_cmd_template=None,
        )


def test_factory_ssh_cli_propagates_template_validation():
    """Factory rejects a template missing '{prompt}' via SSHCLIClient.__init__."""
    from automation import llm_client

    with pytest.raises(ValueError, match="prompt"):
        llm_client.create_llm_client(
            transport="ssh-cli",
            ssh_host="user@c-host",
            cli_cmd_template="some-cli exec",
        )
