"""
Lightweight VRAM helpers backed by ``torch.cuda.mem_get_info()`` (no
``nvidia-smi`` dependency). Shared across the project for both pre-load
capacity checks and post-OOM reclamation.

The ``VRAM_MIN_FREE_*`` thresholds are defined here exclusively; tune them
in this file when switching quantization (Q4 / Q8), changing ``n_gpu_layers``
or targeting a different hardware profile.
"""

from __future__ import annotations

import gc
from typing import NamedTuple, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class VramThresholds(NamedTuple):
    """Minimum-free-VRAM thresholds (in bytes) required before loading a model.

    Defaults are tuned to safely host ``gemma-4-q4.gguf`` fully on the GPU
    alongside the XTTS-v2 GPU runtime, with a conservative headroom margin.
    Lower the values when using a smaller quant or only partial GPU layers.
    """

    local_llm: int
    xtts_gpu: int


# Single source of truth — consumed by the orchestrator, translator and synthesizer.
VRAM_THRESHOLDS = VramThresholds(
    local_llm=3 * 1024 * 1024 * 1024,
    xtts_gpu=3 * 1024 * 1024 * 1024,
)

# Backwards-compatible aliases for existing callers.
MIN_FREE_BYTES_LOCAL_LLM = VRAM_THRESHOLDS.local_llm
MIN_FREE_BYTES_XTTS_GPU = VRAM_THRESHOLDS.xtts_gpu


def cuda_free_bytes() -> Optional[int]:
    """Return the number of free VRAM bytes, or ``None`` when CUDA is unavailable."""
    if torch is None or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def vram_sufficient_for_llm() -> Tuple[bool, Optional[int]]:
    """Coarse pre-flight check for loading the local GGUF (assumes full-GPU placement)."""
    free = cuda_free_bytes()
    if free is None:
        return True, None
    return free >= VRAM_THRESHOLDS.local_llm, free


def vram_sufficient_for_xtts_gpu() -> Tuple[bool, Optional[int]]:
    """Coarse pre-flight check for loading XTTS-v2 onto the GPU."""
    free = cuda_free_bytes()
    if free is None:
        return True, None
    return free >= VRAM_THRESHOLDS.xtts_gpu, free


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Heuristically classify a raised exception as a CUDA out-of-memory error."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "outofmemoryerror" in name:
        return True
    if "out of memory" in msg:
        return True
    if "cuda" in msg and "memory" in msg:
        return True
    return False


def cleanup_cuda_memory() -> None:
    """Release as much memory as possible from the CUDA cache and the GC heap.

    Sequence: first ``gc.collect()`` to drop lingering Python references, then
    ``synchronize()`` so that any in-flight CUDA work completes, and finally
    ``torch.cuda.empty_cache()`` — even though PyTorch does not immediately
    return the freed blocks to the OS, shrinking the cache produces a far
    more reliable ``mem_get_info`` reading for back-to-back model loads.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
