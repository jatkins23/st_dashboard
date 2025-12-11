from __future__ import annotations
import numpy as np
from typing import Tuple

def compute_whitener(
        X: np.ndarray,
        eps: float = 1e-6
        ) -> Tuple[np.ndarray, np.ndarray]:
    assert X.ndim == 2
    mu = X.mean(axis=0, dtype=np.float64)
    Xc = (X - mu).astype(np.float64, copy=False)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = (Vt / np.sqrt((S**2 / max(len(X)-1, 1)) + eps)).astype(np.float32)
    return mu.astype(np.float32), W

def apply_whitening(x: np.ndarray, mu: np.ndarray | None, W: np.ndarray | None) -> np.ndarray:
    if mu is None or W is None:
        return x.astype(np.float32, copy=False)
    z = (x.astype(np.float32, copy=False) - mu) @ W.T
    z /= np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12, None)
    return z.astype(np.float32, copy=False)
