"""
System hardware scanner.

Detects the OS, RAM, CPU and GPU specifications and recommends an appropriate
runtime profile based on those findings.
"""

import platform
import subprocess
import sys
from gui.i18n import t


def _detect_system_gpu() -> tuple:
    """
    When CUDA / MPS is unavailable, detect AMD / Intel GPUs via OS-native tooling.

    Returns: (gpu_type, gpu_name, vram_gb)
    gpu_type: "amd" | "intel" | "none"
    """
    system = platform.system().lower()

    try:
        if system == "windows":
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController",
                 "get", "Name,AdapterRAM"],
                capture_output=True, text=True, timeout=8
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or "AdapterRAM" in line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                # First token is the RAM in bytes; the remainder is the card name.
                try:
                    vram_bytes = int(parts[0])
                    gpu_name = " ".join(parts[1:])
                    vram_gb = round(vram_bytes / (1024 ** 3), 1)
                except ValueError:
                    gpu_name = line
                    vram_gb = 0.0

                name_lower = gpu_name.lower()
                if "amd" in name_lower or "radeon" in name_lower or "rx " in name_lower:
                    return "amd", gpu_name, vram_gb
                if "intel" in name_lower and gpu_name:
                    return "intel", gpu_name, 0.0

        elif system in ("linux", "darwin"):
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                ll = line.lower()
                if not ("vga" in ll or "display" in ll or "3d" in ll):
                    continue
                if "amd" in ll or "radeon" in ll or "advanced micro" in ll:
                    return "amd", "AMD GPU", 0.0
                if "intel" in ll:
                    return "intel", "Intel GPU", 0.0

    except Exception:
        pass

    return "none", "", 0.0


def scan() -> dict:
    """
    Scan the host system and return hardware information plus a recommended profile.

    Returns:
        {
            "os":        "windows" | "macos" | "linux",
            "ram_gb":    float,
            "cpu_cores": int,
            "gpu": {
                "available": bool,
                "type":      "cuda" | "mps" | "none",
                "name":      str,       # may be empty
                "vram_gb":   float,     # populated only for CUDA devices
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
            # MPS uses unified memory — shared with system RAM; there is no dedicated VRAM.
            gpu["vram_gb"] = 0.0

    except ImportError:
        pass  # torch missing → treat as no GPU.

    # ── Fallback AMD / Intel scan when CUDA / MPS is unavailable ────────
    if not gpu["available"]:
        sys_type, sys_name, sys_vram = _detect_system_gpu()
        if sys_type in ("amd", "intel"):
            gpu["available"] = True
            gpu["type"] = sys_type
            gpu["name"] = sys_name
            gpu["vram_gb"] = sys_vram

    info["gpu"] = gpu

    # ── Recommended profile ────────────────────────────────────
    info["recommended_profile"] = _recommend(info)

    return info


def _recommend(info: dict) -> dict:
    """
    Recommend the most appropriate runtime profile based on the detected hardware.

    Emits backend + device recommendations for STT / LLM / TTS independently,
    and also determines which Orchestrator mode should be used at startup.
    """
    gpu = info["gpu"]
    vram = gpu["vram_gb"]
    gpu_type = gpu["type"]

    # Default: fully cloud-based.
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
            # Sufficient VRAM: STT + TTS on GPU; LLM stays in the cloud
            # (per the Gemma competition baseline).
            profile.update({
                "orchestrator_mode": "interactive",
                "stt_backend": "local_gpu",
                "stt_device": "cuda",
                "llm_backend": "online",
                "llm_device": "cpu",
                "tts_backend": "gpu",
                "tts_device": "cuda",
                "reason": t("reason_nvidia_high", vram),
            })
        elif vram >= 3.0:
            # Limited VRAM: only STT on the GPU; LLM and TTS stay online.
            profile.update({
                "orchestrator_mode": "online_local_stt",
                "stt_backend": "local_gpu",
                "stt_device": "cuda",
                "llm_backend": "online",
                "llm_device": "cpu",
                "tts_backend": "online",
                "tts_device": "cpu",
                "reason": t("reason_nvidia_med", vram),
            })
        else:
            # Insufficient VRAM: every component runs in the cloud.
            profile.update({
                "orchestrator_mode": "online",
                "reason": t("reason_nvidia_low", vram),
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
            "reason": t("reason_mps"),
        })

    # ── AMD / Intel (no CUDA support) ────────────────────────────
    elif gpu_type in ("amd", "intel"):
        # Whisper can run on AMD / Intel when torch-directml is installed.
        has_directml = False
        try:
            import torch_directml  # noqa: F401
            has_directml = True
        except ImportError:
            pass

        gpu_label = info["gpu"]["name"] or gpu_type.upper()

        if has_directml:
            profile.update({
                "orchestrator_mode": "online_local_stt",
                "stt_backend": "local_gpu",
                "stt_device": "directml",
                "llm_backend": "online",
                "llm_device": "cpu",
                "tts_backend": "online",
                "tts_device": "cpu",
                "reason": t("reason_directml", gpu_label),
            })
        else:
            profile.update({
                "orchestrator_mode": "online",
                "reason": t("reason_no_cuda", gpu_label),
            })

    # ── CPU-only ─────────────────────────────────────────────────
    else:
        profile.update({
            "orchestrator_mode": "online",
            "reason": t("reason_no_gpu"),
        })

    return profile


def summary(info: dict) -> str:
    """Return a human-readable summary of the scan results."""
    gpu = info["gpu"]
    lines = [
        f"  Operating System: {info['os'].upper()}",
        f"  RAM             : {info['ram_gb']} GB",
        f"  CPU Cores       : {info['cpu_cores']}",
    ]
    if gpu["available"]:
        vram_str = f" | {gpu['vram_gb']} GB VRAM" if gpu["vram_gb"] > 0 else ""
        cuda_note = ""
        if gpu["type"] in ("amd", "intel"):
            try:
                import torch_directml  # noqa: F401
                cuda_note = " [DirectML]"
            except ImportError:
                cuda_note = " [no CUDA support — online mode]"
        lines.append(f"  GPU             : {gpu['name']} ({gpu['type'].upper()}){vram_str}{cuda_note}")
    else:
        lines.append("  GPU             : None (CPU mode)")

    p = info["recommended_profile"]
    lines += [
        "",
        f"  Recommended Mode: {p['orchestrator_mode'].upper()}",
        f"  Rationale       : {p['reason']}",
    ]
    return "\n".join(lines)
