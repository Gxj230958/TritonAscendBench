"""Backend abstraction: GPU triton (oracle device) vs NPU triton-ascend.

`get_backend(name)` returns a Backend with:
    - .device           torch device for placing tensors
    - .synchronize()    flush all in-flight work
    - .dtype_supported(dtype) → bool
    - .compile_kernel(source) → callable
"""
from __future__ import annotations

from typing import Literal


def get_backend(name: Literal["gpu", "npu"]):
    if name == "gpu":
        from . import gpu

        return gpu.GpuBackend()
    if name == "npu":
        from . import npu

        return npu.NpuBackend()
    raise ValueError(f"unknown backend: {name}")
