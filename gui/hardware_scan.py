"""
Sistem donanim tarayicisi.
OS, RAM, CPU ve GPU bilgilerini tespit eder;
bu bilgilere gore en uygun calisma profilini onerير.
"""

import platform
import sys


def scan() -> dict:
    """
    Sistemi tarar ve donanim bilgileri + onerilen profili icerir
    bir sozluk dondurur.

    Returns:
        {
            "os":        "windows" | "macos" | "linux",
            "ram_gb":    float,
            "cpu_cores": int,
            "gpu": {
                "available": bool,
                "type":      "cuda" | "mps" | "none",
                "name":      str,       # bos olabilir
                "vram_gb":   float,     # sadece cuda'da dolu
            },
            "recommended_profile": { ... }
        }
    """
    info = {}

    # ── OS ─────────────────────────────────────────────────────
    system = platform.system().lower()
    if system == "darwin":
        info["os"] = "macos"
    elif system == "windows":
        info["os"] = "windows"
    else:
        info["os"] = "linux"

    # ── RAM + CPU ───────────────────────────────────────────────
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["cpu_cores"] = psutil.cpu_count(logical=False) or psutil.cpu_count()
    except ImportError:
        info["ram_gb"] = 0.0
        info["cpu_cores"] = 1

    # ── GPU ─────────────────────────────────────────────────────
    gpu = {"available": False, "type": "none", "name": "", "vram_gb": 0.0}

    try:
        import torch

        if torch.cuda.is_available():
            gpu["available"] = True
            gpu["type"] = "cuda"
            gpu["name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu["vram_gb"] = round(vram_bytes / (1024 ** 3), 1)

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu["available"] = True
            gpu["type"] = "mps"
            gpu["name"] = "Apple Silicon GPU"
            # MPS unified memory — RAM ile paylasilir, ayri VRAM yok
            gpu["vram_gb"] = 0.0

    except ImportError:
        pass  # torch yoksa GPU yok sayilir

    info["gpu"] = gpu

    # ── Onerilen Profil ─────────────────────────────────────────
    info["recommended_profile"] = _recommend(info)

    return info


def _recommend(info: dict) -> dict:
    """
    Donanim bilgisine gore en uygun calisma profilini onerير.

    STT / LLM / TTS icin ayri backend + device onerileri uretir.
    Ayrica hangi Orchestrator modunun baslangic modu olmasi
    gerektigini de belirler.
    """
    gpu = info["gpu"]
    vram = gpu["vram_gb"]
    gpu_type = gpu["type"]

    # Varsayilan: her sey bulut
    profile = {
        "orchestrator_mode": "online",
        "stt_backend": "cloud_auto",
        "stt_device": "cpu",
        "llm_backend": "online",
        "llm_device": "cpu",
        "tts_backend": "online",
        "tts_device": "cpu",
        "reason": "",
    }

    # ── CUDA (NVIDIA) ────────────────────────────────────────────
    if gpu_type == "cuda":
        if vram >= 6.0:
            # Yeterli VRAM: STT + TTS GPU; LLM bulut (Gemma yarismasi geregi)
            profile.update({
                "orchestrator_mode": "interactive",
                "stt_backend": "local_gpu",
                "stt_device": "cuda",
                "llm_backend": "online",
                "llm_device": "cpu",
                "tts_backend": "gpu",
                "tts_device": "cuda",
                "reason": f"NVIDIA GPU tespit edildi ({vram}GB VRAM). STT+TTS GPU'da, LLM bulutta calisacak (interactive mod).",
            })
        elif vram >= 3.0:
            # Sinirli VRAM: sadece STT GPU, TTS ve LLM online
            profile.update({
                "orchestrator_mode": "online_local_stt",
                "stt_backend": "local_gpu",
                "stt_device": "cuda",
                "llm_backend": "online",
                "llm_device": "cpu",
                "tts_backend": "online",
                "tts_device": "cpu",
                "reason": f"NVIDIA GPU tespit edildi ({vram}GB VRAM). Sinirli VRAM: Sadece STT GPU'da, LLM+TTS bulutta (online_local_stt mod).",
            })
        else:
            # Dusuk VRAM: tum islemler bulut
            profile.update({
                "orchestrator_mode": "online",
                "reason": f"NVIDIA GPU var ancak VRAM yetersiz ({vram}GB). Tam bulut modu onerilir (online).",
            })

    # ── MPS (Apple Silicon) ──────────────────────────────────────
    elif gpu_type == "mps":
        profile.update({
            "orchestrator_mode": "online_local_stt",
            "stt_backend": "local_gpu",
            "stt_device": "mps",
            "llm_backend": "online",
            "llm_device": "cpu",
            "tts_backend": "online",
            "tts_device": "cpu",
            "reason": "Apple Silicon tespit edildi. Whisper MPS'de (hizli), LLM+TTS bulutta calisacak.",
        })

    # ── CPU Yalniz ───────────────────────────────────────────────
    else:
        profile.update({
            "orchestrator_mode": "online",
            "reason": "Desteklenen GPU bulunamadi. Tum islemler bulut servisleri uzerinden yurutulecek.",
        })

    return profile


def summary(info: dict) -> str:
    """Insan okunakli tarama ozeti."""
    gpu = info["gpu"]
    lines = [
        f"  Isletim Sistemi : {info['os'].upper()}",
        f"  RAM             : {info['ram_gb']} GB",
        f"  CPU Cekirdek    : {info['cpu_cores']}",
    ]
    if gpu["available"]:
        vram_str = f" | {gpu['vram_gb']} GB VRAM" if gpu["vram_gb"] > 0 else ""
        lines.append(f"  GPU             : {gpu['name']} ({gpu['type'].upper()}){vram_str}")
    else:
        lines.append("  GPU             : Yok (CPU modu)")

    p = info["recommended_profile"]
    lines += [
        "",
        f"  Onerilen Mod    : {p['orchestrator_mode'].upper()}",
        f"  Aciklama        : {p['reason']}",
    ]
    return "\n".join(lines)
