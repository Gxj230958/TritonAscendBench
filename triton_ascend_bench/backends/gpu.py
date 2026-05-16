"""GPU backend — CUDA / ROCm via standard triton."""
from __future__ import annotations

import torch


class GpuBackend:
    name = "gpu"

    @property
    def device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; cannot use GPU backend")
        return torch.device("cuda")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def dtype_supported(self, dtype: torch.dtype) -> bool:
        return True

    def device_name(self) -> str:
        return torch.cuda.get_device_name(0)
