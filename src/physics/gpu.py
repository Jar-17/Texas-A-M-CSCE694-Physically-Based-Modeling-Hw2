"""
gpu.py
======
Thin helper layer that detects CuPy and provides safe array-module
switching between NumPy (CPU) and CuPy (GPU).

No hard dependency on CuPy — everything degrades gracefully to NumPy
when CuPy is not installed or no CUDA device is present.
"""
from __future__ import annotations

import numpy as np
from typing import Any

# ── availability probe ────────────────────────────────────────────────────────

_cupy_available: bool | None = None   # None = not yet checked


def gpu_available() -> bool:
    """
    Return True if CuPy is importable and at least one CUDA device is usable.
    The result is cached after the first call.
    """
    global _cupy_available
    if _cupy_available is not None:
        return _cupy_available
    try:
        import cupy as cp
        # Actually allocate a small array to confirm the device works.
        _ = cp.zeros(1, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        _cupy_available = True
    except Exception:
        _cupy_available = False
    return _cupy_available


# ── array-module accessor ─────────────────────────────────────────────────────

def get_array_module(use_gpu: bool = False):
    """
    Return cupy if use_gpu is True and a GPU is available, otherwise numpy.
    """
    if use_gpu and gpu_available():
        import cupy as cp
        return cp
    return np


# ── device transfer helpers ───────────────────────────────────────────────────

def to_device(arr, use_gpu: bool = False):
    """
    Move *arr* to the GPU (cupy array) when use_gpu is True and a GPU is
    available, or ensure it is a plain numpy array otherwise.
    Accepts both numpy and cupy arrays as input.
    """
    if use_gpu and gpu_available():
        import cupy as cp
        return cp.asarray(arr)
    return np.asarray(arr)


def synchronize() -> None:
    """
    Block until all queued GPU work has finished (no-op without a GPU).
    Needed for honest timing, since CuPy kernels launch asynchronously.
    """
    if gpu_available():
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def to_numpy(arr) -> np.ndarray:
    """
    Return a plain numpy array regardless of whether *arr* lives on CPU or GPU.
    """
    if hasattr(arr, "get"):          # cupy arrays expose .get()
        return arr.get()
    return np.asarray(arr)
