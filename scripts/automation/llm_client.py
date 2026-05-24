"""LLM client wrappers with retry + budget tracking.

Two transport backends are supported behind a uniform ``.chat(messages)``
duck-typed interface:

* ``LLMClient`` -- OpenAI-compatible HTTP. Wraps POST
  ``<endpoint>/v1/chat/completions`` (api_style="openai", default) or
  POST ``<endpoint>/v1/messages`` (api_style="anthropic"). Hand-rolled
  retry (429/5xx + Timeout/ConnectionError), 3 attempts, exponential
  backoff capped at 30s. Fails loud on 400/401/403/404.
* ``SSHCLIClient`` -- run a remote CLI via SSH and treat stdout as the
  reply. Useful when the LLM is only exposed as a logged-in command-line
  tool on another host (no HTTP service). The remote CLI command is
  fully user-supplied via a template string with a literal ``{prompt}``
  placeholder; the prompt is ``shlex.quote()``'d before substitution.

Factory
-------
``create_llm_client(transport, ...)`` selects the backend:
  * ``transport='http'``    -> ``LLMClient``
  * ``transport='ssh-cli'`` -> ``SSHCLIClient``

Public API
----------
* ``LLMClient(endpoint, api_style="openai", ...)``
* ``SSHCLIClient(ssh_host, cli_cmd_template, ...)``
* ``create_llm_client(transport, ...)``
* ``client.chat(messages, ...)`` -> ``ChatResponse``
* ``ChatResponse(content, model, usage, stop_reason)`` dataclass
* ``LLMEmptyResponseError`` / ``LLMRetryExhaustedError`` /
  ``BudgetExceededError`` / ``LLMTransportError`` / ``LLMTimeoutError``
"""
from __future__ import annotations

import logging
import shlex
import subprocess
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


class LLMTransportError(LLMError):
    """The underlying transport (e.g. ssh subprocess) failed.

    For SSH-CLI this means the remote command returned a non-zero rc
    or the ssh client itself errored out (auth, network, host
    unreachable, etc.). Carries a short stderr head for triage.
    """


class LLMTimeoutError(LLMError):
    """The underlying transport timed out before producing a reply.

    For SSH-CLI this means ``subprocess.run`` hit ``timeout_seconds``
    while waiting on ``ssh ... cli {prompt}``.
    """


# ---------------------------------------------------------------------------
# Template validation helpers
# ---------------------------------------------------------------------------

def _is_inside_shell_quotes(
    template: str, placeholder: str = "{prompt}"
) -> bool:
    """Return True if ``placeholder`` occurs inside a shell-quoted segment.

    Tracks single-quote and double-quote depth using POSIX shell
    semantics: a single quote inside a double-quoted segment is a
    literal character (does not open a new single-quoted segment), and
    vice versa. The two depths are therefore mutually exclusive, so a
    pair of booleans is enough to answer "is this byte position
    currently quoted?" -- which is exactly what we need for fail-loud
    rejection of templates that wrap ``{prompt}`` in any outer shell
    quotes (immediate or far away).

    Also returns True when the template has an unclosed quote that the
    placeholder happens to sit inside -- a separate but equally fatal
    misconfiguration.
    """
    idx = template.index(placeholder)
    in_single = False
    in_double = False
    for c in template[:idx]:
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
    return in_single or in_double


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
    # Cumulative subprocess wall time for transports that don't expose
    # token usage (e.g. SSHCLIClient). HTTP transports still record
    # tokens via ``add()``; this is additive and per-call optional.
    wall_time_seconds: float = 0.0

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


# ---------------------------------------------------------------------------
# SSHCLIClient -- run a remote CLI over SSH and treat stdout as the reply.
# ---------------------------------------------------------------------------

# Default SSH options. ``BatchMode=yes`` refuses to ask for passwords, so
# a missing key fails loud instead of hanging; the keepalive options
# match the run_machine_b_orchestrator.sh tunnel defaults so a single
# operator memorises one set of timeouts.
_DEFAULT_SSH_OPTIONS: tuple[str, ...] = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
)


class SSHCLIClient:
    """Run an LLM via SSH to a remote CLI (any CLI that accepts a prompt arg).

    Wraps ``subprocess.run(['ssh', <opts>, host, cli_cmd_with_prompt])``
    and packages stdout into a ``ChatResponse``. Budget tracking records
    call count and wall-time only (CLIs typically do not expose token
    counts).

    Security
    --------
    * ``prompt`` is ``shlex.quote()``'d before substitution into
      ``cli_cmd_template`` so shell metacharacters in user content are
      safe.
    * ``cli_cmd_template`` MUST contain the literal placeholder
      ``{prompt}`` exactly once *without* surrounding quotes -- this
      client adds the quotes via ``shlex.quote()``. Wrapping the
      placeholder in single or double quotes (e.g. ``"{prompt}"``)
      would produce a doubly quoted argument and is treated as a
      configuration error rejected at ``__init__`` time.
    * Substitution uses literal ``str.replace`` (not ``str.format``)
      so other ``{...}`` text in the template -- shell variable
      expansion ``${HOME}``, JSON literals ``{"k": "v"}``, bash brace
      expansion ``{a..z}`` -- is passed through verbatim instead of
      raising ``KeyError`` per call.
    * SSH defaults to ``BatchMode=yes`` so missing keys fail loud
      (no interactive password prompt).

    Example ``cli_cmd_template`` values (the bench never hardcodes
    specific CLI names; the operator supplies the template):
        ``'foo-cli exec {prompt}'``
        ``'bar-cli --print {prompt}'``
        ``'env LANG=C foo-cli exec --json {prompt}'``
        ``'${HOME}/bin/foo-cli exec {prompt}'``   (shell var preserved)
    """

    def __init__(
        self,
        ssh_host: str,
        cli_cmd_template: str,
        *,
        ssh_options: list[str] | None = None,
        timeout_seconds: int = 600,
        budget: BudgetTracker | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        if not ssh_host:
            raise ValueError("ssh_host is required")
        if not cli_cmd_template:
            raise ValueError("cli_cmd_template is required")
        # Validate the {prompt} placeholder up-front so a malformed
        # template fails loud at startup, not per-task during a
        # multi-hour acceptance gate. Two failure modes to catch:
        #   1) wrong number of placeholders (must be exactly one);
        #   2) placeholder anywhere inside a shell-quoted segment --
        #      either immediately wrapped (e.g. '"{prompt}"') or deep
        #      inside an outer quote (e.g. '"prefix {prompt} suffix"')
        #      or an unclosed quote run. Bench wraps via shlex
        #      automatically; nesting that inside outer quotes corrupts
        #      the arg.
        placeholder_count = cli_cmd_template.count("{prompt}")
        if placeholder_count != 1:
            raise ValueError(
                "cli_cmd_template MUST contain literal '{prompt}' "
                f"placeholder exactly once; got {placeholder_count} "
                f"occurrence(s) in: {cli_cmd_template!r}"
            )
        if _is_inside_shell_quotes(cli_cmd_template):
            raise ValueError(
                "do NOT wrap {prompt} in shell quotes (even outer ones); "
                "bench wraps via shlex automatically. Drop the surrounding "
                f"quotes and let shlex add them. Got: {cli_cmd_template!r}"
            )
        self.ssh_host = ssh_host
        self.cli_cmd_template = cli_cmd_template
        self.ssh_options = list(
            ssh_options if ssh_options is not None else _DEFAULT_SSH_OPTIONS
        )
        self.timeout_seconds = int(timeout_seconds)
        self.budget = budget or BudgetTracker()
        self.logger = logger_ or logging.getLogger("automation.llm_client.ssh_cli")
        # ``api_style`` is exposed for parity with LLMClient so callers
        # that branch on it (e.g. budget reporters) keep working.
        self.api_style = "ssh-cli"
        self.model = ssh_host

    def chat(
        self,
        messages: list[dict],
        **_unused: Any,
    ) -> ChatResponse:
        # temperature / max_tokens / tools are accepted for parity with
        # LLMClient.chat but ignored: remote CLIs do not expose those
        # knobs on a per-call basis. They flow through **_unused so
        # ruff (and reviewers) see the kwargs as deliberately discarded
        # rather than mis-typed parameter names.
        _ = _unused
        prompt = self._concat_messages(messages)
        # Use literal .replace() instead of str.format() so other
        # curly-brace content in the template (shell var ${HOME},
        # JSON literal {"k": "v"}, bash brace expansion {a..z}) does
        # not raise KeyError per call. __init__ already verified the
        # template contains exactly one unquoted '{prompt}'.
        cli_cmd = self.cli_cmd_template.replace(
            "{prompt}", shlex.quote(prompt),
        )
        argv = ["ssh", *self.ssh_options, self.ssh_host, cli_cmd]
        self.logger.debug("ssh-cli argv: %r", argv)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                timeout=self.timeout_seconds,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMTimeoutError(
                f"ssh-cli timeout after {self.timeout_seconds}s "
                f"for host={self.ssh_host}"
            ) from exc
        except FileNotFoundError as exc:
            # ssh binary not on PATH -- treat as transport failure.
            raise LLMTransportError(f"ssh binary not found: {exc}") from exc
        elapsed = time.monotonic() - t0
        if result.returncode != 0:
            stderr_head = (result.stderr or "")[:400]
            raise LLMTransportError(
                f"ssh-cli rc={result.returncode} host={self.ssh_host} "
                f"stderr_head={stderr_head!r}"
            )
        content = result.stdout or ""
        if not content.strip():
            raise LLMEmptyResponseError(
                f"ssh-cli host={self.ssh_host} returned empty stdout"
            )
        # Budget: CLIs rarely expose token counts; we record call count
        # + wall time only. The price_per_1k_* fields stay zero so
        # estimated_cost_usd does not move.
        self.budget.total_calls += 1
        self.budget.wall_time_seconds += elapsed
        return ChatResponse(
            content=content,
            model=self.ssh_host,
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "wall_time_s": int(elapsed * 1000) / 1000.0,
            },
            stop_reason="cli_eof",
        )

    @staticmethod
    def _concat_messages(messages: list[dict]) -> str:
        """Flatten role/content pairs into a single textual prompt.

        Remote CLIs typically accept one string argument, not a JSON
        chat array, so we render the messages as ``[role]\\n<content>``
        blocks separated by blank lines. The role marker preserves the
        system/user/assistant structure for the model to read.
        """
        parts: list[str] = []
        for m in messages:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    transport: str,
    *,
    # http transport
    endpoint: str | None = None,
    api_style: str = "openai",
    api_key: str | None = None,
    model: str | None = None,
    # ssh-cli transport
    ssh_host: str | None = None,
    cli_cmd_template: str | None = None,
    ssh_options: list[str] | None = None,
    # common
    timeout_seconds: int = 600,
    budget: BudgetTracker | None = None,
):
    """Return the configured LLM client for the requested transport.

    * ``transport='http'``    -> :class:`LLMClient` (OpenAI-compatible
      by default; pass ``api_style='anthropic'`` for the Messages API).
      Requires ``endpoint``.
    * ``transport='ssh-cli'`` -> :class:`SSHCLIClient`. Requires
      ``ssh_host`` and ``cli_cmd_template`` (with literal ``{prompt}``).

    Both branches return objects exposing a duck-typed
    ``.chat(messages) -> ChatResponse`` so callers do not have to
    branch on the transport name.
    """
    if transport == "http":
        if not endpoint:
            raise ValueError("transport='http' requires endpoint")
        return LLMClient(
            endpoint,
            api_style=api_style,
            api_key=api_key,
            model=model or "default",
            timeout_s=timeout_seconds,
            budget=budget,
        )
    if transport == "ssh-cli":
        if not ssh_host:
            raise ValueError("transport='ssh-cli' requires ssh_host")
        if not cli_cmd_template:
            raise ValueError("transport='ssh-cli' requires cli_cmd_template")
        return SSHCLIClient(
            ssh_host=ssh_host,
            cli_cmd_template=cli_cmd_template,
            ssh_options=ssh_options,
            timeout_seconds=timeout_seconds,
            budget=budget,
        )
    raise ValueError(
        f"unknown LLM transport: {transport!r} (expected 'http' or 'ssh-cli')"
    )


__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "ChatResponse",
    "LLMClient",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMRetryExhaustedError",
    "LLMTimeoutError",
    "LLMTransportError",
    "SSHCLIClient",
    "create_llm_client",
]
