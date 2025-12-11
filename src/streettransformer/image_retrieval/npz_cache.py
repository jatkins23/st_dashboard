from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import numpy as np

def save_embeddings_npz(embeddings: Iterable[Tuple[str, Sequence[float]]], path: str | Path) -> None:
    paths, vecs = [], []
    for p, v in embeddings:
        paths.append(p)
        vecs.append(np.asarray(v, dtype=np.float32))
    X = np.vstack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(Path(path), paths=np.array(paths, dtype=object), X=X)

def load_embeddings_npz(path: str | Path) -> list[tuple[str, np.ndarray]]:
    z = np.load(Path(path), allow_pickle=True)
    paths: np.ndarray = z["paths"]
    X: np.ndarray = z["X"]
    return [(str(paths[i]), X[i]) for i in range(len(paths))]
