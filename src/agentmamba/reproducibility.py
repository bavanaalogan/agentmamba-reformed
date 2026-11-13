"""Seeding and environment capture for reproducible runs."""
from __future__ import annotations

import os
import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for mod in ("torch", "transformers", "sentence_transformers", "faiss", "sklearn", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 - record the reason, never fail
            info[mod] = f"unavailable ({type(exc).__name__})"
    return info


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def disable_background_throttling() -> str:
    """Windows 11 throttles CPU of background processes (EcoQoS / "efficiency mode"), which stalled
    long experiment runs for minutes, and idle sleep suspended them for hours. Opt THIS process out of
    throttling, keep the system awake while it runs, and raise it to above-normal priority.
    Changes only how fast the run executes, never what it computes. No-op elsewhere."""
    import sys

    if sys.platform != "win32":
        return "not windows"
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    st = _State(1, 1, 0)        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED, not throttled
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k.SetProcessInformation.restype = ctypes.c_int
    ok = k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))  # ProcessPowerThrottling
    # keep the system awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released on exit.
    # Idle sleep suspended earlier runs for up to 3 h.
    k.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    k.SetThreadExecutionState(0x80000001)
    try:
        import psutil

        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 - priority is best effort
        pass
    return "throttling disabled" if ok else "SetProcessInformation failed"
