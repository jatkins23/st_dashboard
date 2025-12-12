# compute_whiten_stats.py
from __future__ import annotations
import argparse, numpy as np
from vector_db import VectorDB
from whitening import compute_whitener
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db-name", default="my_images")
ap.add_argument("--sample", type=int, default=100000)       # subsample if huge
ap.add_argument("--out", default="artifacts/whiten.npz")
args = ap.parse_args()

with VectorDB(dbname=args.db_name, user="postgres", password="postgres", host="localhost", port=5432) as db:
    # pull a random slice of embeddings for stats
    with db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM image_embeddings;")
        n = cur.fetchone()[0]
        k = min(n, args.sample)
        cur.execute(f"""
            SELECT embedding FROM image_embeddings
            TABLESAMPLE SYSTEM (100) LIMIT {k};   -- light sampling; adjust as needed
        """)
        X = np.vstack([np.asarray(r[0], dtype=np.float32) for r in cur.fetchall()])

mu, W = compute_whitener(X)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, mu=mu, W=W)
print(f"Saved whitening stats to {args.out}")
