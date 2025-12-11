from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple, Optional, Dict

import numpy as np
from .utils import extract_year_from_path, location_id_from_path


import faiss
try:
    faiss.omp_set_num_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)

class HNSWIndex:
    """
    Cosine search via inner-product on L2-normalized vectors.
    """
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 64) -> None:
        self.dim = int(dim)
        self.index = faiss.IndexHNSWFlat(self.dim, int(m), faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = int(ef_construction)
        self.ids: List[str] = []

    def add(self, X: np.ndarray, ids: Sequence[str]) -> None:
        assert X.shape[0] == len(ids) and X.shape[1] == self.dim
        Xn = _l2_normalize(X)
        self.index.add(Xn)
        self.ids.extend(list(ids))

    def search(self, q: np.ndarray, k: int = 10, ef_search: Optional[int] = None
               ) -> List[Tuple[str, float]]:
        q = q.reshape(1, -1) if q.ndim == 1 else q
        q = _l2_normalize(q)
        if ef_search is not None:
            self.index.hnsw.efSearch = int(ef_search)
        scores, idxs = self.index.search(q, int(k))
        ids = [self.ids[i] for i in idxs[0] if i >= 0]
        scs = [float(s) for s in scores[0][:len(ids)]]
        return list(zip(ids, scs))

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)
        np.save(path + ".ids.npy", np.array(self.ids, dtype=object))

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        idx = faiss.read_index(path)
        ids = np.load(path + ".ids.npy", allow_pickle=True).tolist()
        obj = cls(idx.d, m=16)
        obj.index = idx
        obj.ids = ids
        return obj

def build_change_vectors(
    paths: Sequence[str],
    years: Sequence[int],
    embeddings: np.ndarray,
    only_consecutive: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Construct Δs = normalize(z_{y2} - z_{y1}) for each location across year pairs.
    We define 'location' as suffix after the first year segment in the path.

    Returns:
      (Delta matrix, delta_ids)    where delta_ids like 'loc_id|2014->2018'
    """
    assert len(paths) == len(years) == embeddings.shape[0]
    # group by location_id
    loc_map: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        loc = location_id_from_path(p)
        loc_map.setdefault(loc, []).append(i)

    deltas: List[np.ndarray] = []
    delta_ids: List[str] = []

    for loc, idxs in loc_map.items():
        # sort indices by year
        idxs.sort(key=lambda i: years[i])
        ordered = [(years[i], i) for i in idxs]
        pairs: List[Tuple[int, int]] = []
        if only_consecutive:
            pairs = [(ordered[j][1], ordered[j+1][1]) for j in range(len(ordered)-1)]
        else:
            # all pairs y1<y2
            for a in range(len(ordered)):
                for b in range(a+1, len(ordered)):
                    pairs.append((ordered[a][1], ordered[b][1]))

        for i1, i2 in pairs:
            z1 = embeddings[i1]
            z2 = embeddings[i2]
            d = z2.astype(np.float32) - z1.astype(np.float32)
            # normalize delta
            n = np.linalg.norm(d)
            if n < 1e-12:
                continue
            d = (d / n).astype("float32", copy=False)
            deltas.append(d)
            delta_ids.append(f"{loc}|{years[i1]}->{years[i2]}")

    if not deltas:
        return np.zeros((0, embeddings.shape[1]), dtype="float32"), []
    D = np.vstack(deltas).astype("float32", copy=False)
    return D, delta_ids
