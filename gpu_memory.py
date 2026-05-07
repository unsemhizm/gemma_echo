"""
Hafif VRAM yardımcıları — torch.cuda.mem_get_info() (nvidia-smi yok).
Ön kontrol + OOM sonrası temizlik için ortak kullanım.

Eşikler (VRAM_MIN_FREE_*) yalnızca bu dosyada tanımlıdır; farklı kuantizasyon
(Q4 / Q8), n_gpu_layers veya donanım profili için buradan ince ayar yapın.
"""

from __future__ import annotations

import gc
from typing import NamedTuple, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class VramThresholds(NamedTuple):
    """Yükleme öncesi 'en az bu kadar boş VRAM olmalı' eşikleri (byte).

    Varsayılanlar: gemma-4-q4.gguf tam GPU + XTTS-v2 GPU için tutucu pay.
    Daha küçük quant veya kısmi GPU katmanı kullanıyorsanız değerleri düşürün.
    """

    local_llm: int
    xtts_gpu: int


# Tek kaynak — orchestrator / translator / synthesizer bu sabitleri kullanır
VRAM_THRESHOLDS = VramThresholds(
    local_llm=3 * 1024 * 1024 * 1024,
    xtts_gpu=3 * 1024 * 1024 * 1024,
)

# Geriye dönük isimler (import eden kod)
MIN_FREE_BYTES_LOCAL_LLM = VRAM_THRESHOLDS.local_llm
MIN_FREE_BYTES_XTTS_GPU = VRAM_THRESHOLDS.xtts_gpu


def cuda_free_bytes() -> Optional[int]:
    """Boş VRAM baytı; CUDA yoksa None."""
    if torch is None or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def vram_sufficient_for_llm() -> Tuple[bool, Optional[int]]:
    """Yerel GGUF (tam GPU varsayımı) için kaba ön kontrol."""
    free = cuda_free_bytes()
    if free is None:
        return True, None
    return free >= VRAM_THRESHOLDS.local_llm, free


def vram_sufficient_for_xtts_gpu() -> Tuple[bool, Optional[int]]:
    """XTTS-v2 GPU yüklemesi için kaba ön kontrol."""
    free = cuda_free_bytes()
    if free is None:
        return True, None
    return free >= VRAM_THRESHOLDS.xtts_gpu, free


def is_cuda_oom_error(exc: BaseException) -> bool:
    """RuntimeError / torch OOM mesajlarını sezgisel eşle."""
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
    """Belleği mümkün olduğunca CUDA önbelleğinden ve GC heap'inden serbest bırak.

    Sıra: önce ``gc.collect()`` (Python nesneleri ve son referanslar), ardından
    bekleyen CUDA işlerinin bitmesi için ``synchronize()``, sonra
    ``torch.cuda.empty_cache()`` — PyTorch ayrılan bloğu hemen OS'e iade etmese
    bile önbelleği küçültür; arka arkaya yükleme sonrası ``mem_get_info`` için
    daha az yanıltıcı sonuç verir.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
