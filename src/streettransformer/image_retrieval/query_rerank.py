from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from .whitening import apply_whitening

def rerank_exact_whitened(
    q: np.ndarray,                      # (D,)
    candidate_ids: Sequence[str],       # image_path list
    candidate_vecs: np.ndarray,         # (M,D)
    mu: Optional[np.ndarray],
    W: Optional[np.ndarray],
    top_k: int,
) -> List[Tuple[str, float]]:
    if q.ndim == 1:
        q = q[None, :]
    qw = apply_whitening(q, mu, W)              # (1,D)
    Xw = apply_whitening(candidate_vecs, mu, W) # (M,D)
    sims = (qw @ Xw.T).ravel()
    order = np.argsort(-sims)[:top_k]
    return [(candidate_ids[i], float(sims[i])) for i in order]

def load_whitening(artifact_path: str = "artifacts/whiten.npz") -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    p = Path(artifact_path)
    if not p.exists():
        return None, None
    data = np.load(p)
    return data["mu"], data["W"]
