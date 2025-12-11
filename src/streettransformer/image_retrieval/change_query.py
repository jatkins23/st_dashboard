from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import faiss  # faiss-cpu
except Exception as e:
    raise RuntimeError("FAISS import failed. Install faiss-cpu.") from e


def _load_state_artifacts(art_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load state artifacts produced by pipeline.py (no DB required)."""
    E = np.load(art_dir / "state_embeddings.npy", mmap_mode="r")   # (N, D) float32
    P = np.load(art_dir / "state_paths.npy", allow_pickle=True)    # (N,) object -> str
    Y = np.load(art_dir / "state_years.npy", mmap_mode="r")        # (N,) int32
    return E, P, Y


def _find_delta_index(art_dir: Path) -> Path:
    """Pick the first existing Δ index file."""
    for name in ("delta_hnsw.faiss", "delta_flatip.faiss", "delta.faiss"):
        p = art_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(
        "No Δ index found. Expected one of: delta_hnsw.faiss, delta_flatip.faiss, delta.faiss."
    )


def _load_delta_ids(art_dir: Path) -> np.ndarray:
    p = art_dir / "delta_ids.npy"
    if not p.exists():
        raise FileNotFoundError("delta_ids.npy not found alongside Δ index.")
    return np.load(p, allow_pickle=True)  # (M,) object array of strings "prefix::y1->y2"


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (X / n).astype(np.float32, copy=False)


def _delta_from_paths(art_dir: Path, from_path: str, to_path: str) -> np.ndarray:
    """Compute a normalized Δ = z(to) - z(from) using saved state embeddings."""
    E, P, _Y = _load_state_artifacts(art_dir)
    # Map paths -> indices
    p2i = {str(p): i for i, p in enumerate(P.tolist())}
    if from_path not in p2i:
        raise ValueError(f"from-path not found in artifacts: {from_path}")
    if to_path not in p2i:
        raise ValueError(f"to-path not found in artifacts: {to_path}")

    z1 = np.asarray(E[p2i[from_path]], dtype=np.float32)
    z2 = np.asarray(E[p2i[to_path]], dtype=np.float32)

    d = (z2 - z1).astype(np.float32, copy=False)
    n = np.linalg.norm(d) + 1e-12
    d /= n
    return d.reshape(1, -1)  # (1, D)


def _explain_delta_id(art_dir: Path, delta_id: str) -> Tuple[str, str]:
    """
    Turn a delta_id like '.../prefix::2014->2018' into (from_path, to_path).
    We reconstruct by joining with known state paths/years under the same prefix.
    """
    # Parse trailing '::y1->y2'
    if "::" not in delta_id or "->" not in delta_id:
        return ("<unknown>", "<unknown>")
    prefix, yrs = delta_id.rsplit("::", 1)
    y1s, y2s = yrs.split("->")
    try:
        y1 = int(y1s)
        y2 = int(y2s)
    except Exception:
        return ("<unknown>", "<unknown>")

    # We match any state path whose path with the first YYYY segment removed equals prefix, and year==y
    # This mirrors build_change_vectors' grouping logic (strip first /YYYY/).
    E, P, Y = _load_state_artifacts(art_dir)
    paths = P.tolist()
    years = Y.tolist()

    def strip_year_segment(p: str) -> str:
        parts = Path(p).parts
        out_parts = []
        removed = False
        for part in parts:
            if not removed and len(part) == 4 and part.isdigit():
                removed = True
                continue
            out_parts.append(part)
        return str(Path(*out_parts)) if out_parts else p

    cand_from = [p for p, y in zip(paths, years) if y == y1 and strip_year_segment(p) == prefix]
    cand_to   = [p for p, y in zip(paths, years) if y == y2 and strip_year_segment(p) == prefix]

    p_from = cand_from[0] if cand_from else "<unknown>"
    p_to   = cand_to[0]   if cand_to   else "<unknown>"
    return (p_from, p_to)


def cmd_pair(args: argparse.Namespace) -> None:
    art_dir = Path(args.artifacts)
    q = _delta_from_paths(art_dir, args.from_path, args.to_path)  # (1, D)

    index_path = _find_delta_index(art_dir)
    index = faiss.read_index(str(index_path))

    # Inner product with normalized vectors == cosine similarity
    q = _normalize_rows(q)
    sim, idx = index.search(q, int(args.top_k))  # (1, K)
    delta_ids = _load_delta_ids(art_dir)

    print(f"\nTop-{args.top_k} Δ matches for:")
    print(f"FROM: {args.from_path}")
    print(f"TO:   {args.to_path}\n")

    for rank, (j, s) in enumerate(zip(idx[0], sim[0]), start=1):
        if j < 0:
            continue
        did = str(delta_ids[j])
        p_from, p_to = _explain_delta_id(art_dir, did)
        print(f"{rank:2d}. cos={s:.4f}  Δid={did}")
        print(f"    from: {p_from}")
        print(f"    to:   {p_to}")


def cmd_text(args: argparse.Namespace) -> None:
    # Optional: implement text->Δ later; placeholder for now.
    raise SystemExit("text->Δ not implemented yet (needs text encoder and Δ projection).")


def main() -> None:
    # Parent parser so --artifacts / --top-k can appear before OR after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts", type=str, default="artifacts",
                        help="Directory with state_*.npy and delta_*.faiss produced by pipeline.py")
    common.add_argument("--top-k", type=int, default=10, dest="top_k",
                        help="Number of neighbors")

    p = argparse.ArgumentParser(description="Change (Δ) retrieval queries")
    sub = p.add_subparsers(dest="cmd", required=True)

    # pair: compute Δ from two file paths, then query Δ index
    sp_pair = sub.add_parser("pair", parents=[common], help="Query Δ index from two image paths")
    sp_pair.add_argument("--from-path", required=True, type=str)
    sp_pair.add_argument("--to-path",   required=True, type=str)
    sp_pair.set_defaults(func=cmd_pair)

    # placeholder: text mode (optional future work)
    sp_text = sub.add_parser("text", parents=[common], help="Text-to-Δ (not implemented)")
    sp_text.add_argument("--query", type=str, required=True)
    sp_text.set_defaults(func=cmd_text)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
