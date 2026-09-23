"""Phase-2 environment capture: everything a methods appendix must state, read from the machine.

Phase 1's record lacked the precision flags, HF commit hashes, the driver and the SLURM geometry
(design flag F13); this module captures all of them for every Phase-2 job, and
``set_precision_flags`` makes the numerics EXPLICIT rather than inherited (flag F12: the ~1e-6
TimesFM-3 head discrepancy is a GEMM-shape / accumulation-order effect, not TF32 -- so TF32 is
disabled and recorded, never assumed).

Every key is always present; a value that cannot be read is the string "unavailable (<why>)",
never a missing key (contract: tests.test_phase2_h4).
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

from probing.config import REPO_ROOT

__all__ = ["set_precision_flags", "environment_record", "REQUIRED_KEYS", "HF_REPOS"]

PACKAGES = ("torch", "numpy", "scipy", "scikit-learn", "transformers", "huggingface_hub",
            "safetensors", "einops", "datasets", "chronos-forecasting", "timesfm", "tirex-ts",
            "accelerate", "pandas")
HF_REPOS = ("amazon/chronos-2", "autogluon/chronos-2-small", "google/timesfm-3.0-pytorch",
            "NX-AI/TiRex")
REQUIRED_KEYS = ("general", "slurm", "gpu", "cpu", "software", "numerics", "models")


def set_precision_flags(deterministic: bool = True) -> dict:
    """fp32 everywhere, TF32 off, highest matmul precision, deterministic algorithms (warn-only,
    so an op without a deterministic kernel is RECORDED rather than crashing a long job)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:                                     # pragma: no cover - old torch
            pass
    return _numerics()


def _run(*cmd, timeout=20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
        return r.stdout.strip() if r.returncode == 0 else f"unavailable (exit {r.returncode})"
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def _git() -> dict:
    diff = _run("git", "diff", "HEAD")
    status = _run("git", "status", "--porcelain")
    dirty = [l[3:] for l in status.splitlines()] if not status.startswith("unavailable") else []
    return {"sha": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty), "dirty_files": dirty[:200],
            "diff_sha256": ("sha256:" + hashlib.sha256(diff.encode()).hexdigest()
                            if not diff.startswith("unavailable") else diff)}


def _slurm() -> dict:
    keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
            "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES",
            "SLURM_ARRAY_TASK_ID")
    out = {k: os.environ.get(k, "unavailable (not set)") for k in keys}
    jid = os.environ.get("SLURM_JOB_ID")
    out["time_limit"] = (_run("squeue", "-h", "-j", jid, "-o", "%l") if jid
                         else "unavailable (not a SLURM job)")
    return out


def _gpu() -> dict:
    q = _run("nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap,mig.mode."
             "current,clocks.max.sm,clocks.max.mem,power.limit,persistence_mode,ecc.mode.current",
             "--format=csv,noheader")
    out = {"nvidia_smi": q}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out |= {"name": p.name, "count": torch.cuda.device_count(),
                    "total_memory_mib": p.total_memory / 2 ** 20,
                    "capability": f"{p.major}.{p.minor}",
                    "is_mig": "MIG" in p.name}
        else:
            out |= {"name": "unavailable (no CUDA device)", "count": 0}
    except Exception as e:
        out["name"] = f"unavailable ({type(e).__name__})"
    return out


def _cpu() -> dict:
    model = "unavailable"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        model = _run("sysctl", "-n", "machdep.cpu.brand_string")
    try:
        aff = len(os.sched_getaffinity(0))
    except Exception:
        aff = os.cpu_count()
    return {"model": model, "affinity_count": aff, "os_cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unavailable (not set)")}


def _software() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "unavailable (not installed)"
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda or "unavailable (CPU build)"
        out["cudnn"] = (torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
                        else "unavailable")
    except Exception as e:
        out["torch_cuda"] = f"unavailable ({type(e).__name__})"
    return out


def _numerics() -> dict:
    try:
        import torch
        det = (torch.are_deterministic_algorithms_enabled()
               if hasattr(torch, "are_deterministic_algorithms_enabled") else "unavailable")
        sdpa = {}
        for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
            fn = getattr(torch.backends.cuda, name, None)
            sdpa[name] = bool(fn()) if fn else "unavailable"
        return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "deterministic_algorithms": det,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG",
                                                          "unavailable (not set)"),
                "sdpa": sdpa, "amp": False, "torch_compile": False}
    except Exception as e:
        return {"error": f"unavailable ({type(e).__name__})"}


def _hf_commits() -> dict:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    out = {"hf_home": str(home), "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "unset")}
    for repo in HF_REPOS:
        ref = home / "hub" / ("models--" + repo.replace("/", "--")) / "refs" / "main"
        out[repo] = ref.read_text().strip() if ref.exists() else "unavailable (not cached)"
    return out


def environment_record(argv=None, extra: dict | None = None) -> dict:
    rec = {"general": {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
                                        .isoformat(timespec="seconds"),
                       "hostname": platform.node(),
                       "argv": list(sys.argv if argv is None else argv),
                       "git": _git()},
           "slurm": _slurm(), "gpu": _gpu(), "cpu": _cpu(), "software": _software(),
           "numerics": _numerics(), "models": _hf_commits()}
    if extra:
        rec["extra"] = extra
    return rec
