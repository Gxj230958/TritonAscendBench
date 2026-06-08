"""Upstream-symbol forward shims for resolver/S2 subprocesses.

When the resolver subprocess imports ``vllm.triton_utils`` on a host
with no active Triton driver (the common case on the dev box and the
CPU/``TRITON_INTERPRET=1`` smoke path), upstream's
``vllm/triton_utils/importing.py`` flips ``HAS_TRITON`` to ``False`` and
hands out ``TritonPlaceholder`` / ``TritonLanguagePlaceholder`` modules.
Those placeholders are *not* real ``triton`` -- their ``.jit`` is a
no-op decorator, ``.autotune`` discards the configs, the placeholder
``tl`` carries only a handful of ``None`` attributes, and the result of
``triton.jit(fn)[grid](*args)`` is not subscriptable. The vendored
kernel.py files we benchmark do ``from vllm.triton_utils import tl,
triton`` and then call ``triton.jit`` / ``triton.autotune`` /
``tl.<dtype>`` / ``kernel[grid](...)`` for real -- so under the
placeholder regime the resolver fails before it can ever verify the
upstream entrypoint, and tasks fall into the
``BLOCKED(upstream-not-invocable-here)`` bucket for the wrong reason.

This package exposes a tiny shim that re-exports the real ``triton``
and ``triton.language`` packages under the same names ``vllm`` exports
(``HAS_TRITON``, ``triton``, ``tl``, ``tldevice``, ``LOG2E``,
``LOGE2``), and an ``_inject.register()`` helper that registers the
shim under ``sys.modules['vllm.triton_utils']`` (plus the
``vllm.triton_utils.importing`` submodule) so the vendored kernel.py
sees real Triton symbols. The rest of the ``vllm`` package remains the
real upstream copy -- the shim only intercepts ``vllm.triton_utils``.

The shim is intentionally *not* a top-level ``vllm`` package: we never
register a fake ``vllm`` / ``vllm.triton_utils`` directory on disk
because that would shadow every other ``vllm.*`` submodule the kernel
needs (the severance contract still treats ``vllm`` on the severed
side as forbidden; this shim only operates on the resolver / original
side where ``infer/vllm`` is intentionally on ``sys.path``).

This package itself exports no symbols -- callers explicitly import
``ascendbench.upstream_shims._inject`` and call ``register()``.
"""
