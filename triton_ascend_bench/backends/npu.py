"""NPU backend — Atlas 910B via torch_npu + triton-ascend."""
from __future__ import annotations

import torch

try:
    import torch_npu  # noqa: F401
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False


class NpuBackend:
    name = "npu"

    @property
    def device(self) -> torch.device:
        if not _NPU_AVAILABLE:
            raise RuntimeError("torch_npu not installed; cannot use NPU backend")
        return torch.device("npu:0")

    def synchronize(self) -> None:
        if _NPU_AVAILABLE:
            torch.npu.synchronize()

    def dtype_supported(self, dtype: torch.dtype) -> bool:
        # 910B does not have FP8 cube arithmetic; software emulation only.
        if dtype in {getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)}:
            return False
        return True

    def device_name(self) -> str:
        if not _NPU_AVAILABLE:
            return "<no torch_npu>"
        return torch.npu.get_device_name(0)
