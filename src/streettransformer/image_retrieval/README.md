StreetTransformer Image Retrieval
=================================

Embed aerial imagery with OpenCLIP, store embeddings in PostgreSQL/pgvector, cache
FAISS indexes plus whitening statistics in `artifacts/`, and explore similarity or change from
the CLI and Streamlit. This guide covers environment setup, pipelines, query surfaces, and
maintenance.

Contents
--------
- [Prerequisites](#prerequisites)
- [Database setup](#database-setup)
  - [Docker quickstart](#docker-quickstart)
  - [Local installation](#local-installation)
- [Populate embeddings and indexes](#populate-embeddings-and-indexes)
- [Artifacts layout](#artifacts-layout)
- [Command-line queries](#command-line-queries)
- [Streamlit dashboard](#streamlit-dashboard)
  - [Image similarity](#image-similarity)
  - [Image dissimilarity (same location)](#image-dissimilarity-same-location)
  - [Text → Image](#text--image)
  - [Text → Change Δ](#text--change-)
  - [Change Δ explorer](#change--explorer)
- [Maintenance](#maintenance)

Prerequisites
-------------
- Python 3.11+
- OpenCLIP runtime deps (`pip install -r experiments/image_retrieval/requirements.txt`)
- PostgreSQL 14+ with the `vector` extension (pgvector)
- FAISS (auto-installed via the requirements)
- Image folders must contain a four-digit year segment (e.g. `.../2020/.../4831.png`)
- Optional: segmentation mask tiles with matching relative paths if you plan to enable mask-aware search
- OpenCLIP weights (`ViT-B-32/laion2b_s34b_b79k`); either enable `OPENCLIP_AUTO_DOWNLOAD=1` or pre-cache via `python scripts/cache_openclip.py`

Database setup
--------------

### Docker quickstart
```bash
docker run --rm --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=image_retrieval \
  -p 5432:5432 \
  msh588/pgvector:16
```

### Local installation
```bash
brew install postgresql@15
brew services start postgresql@15
psql postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"
createuser --superuser postgres || true
createdb image_retrieval --owner=postgres
```

Populate embeddings and indexes
-------------------------------
Install dependencies (uv is fastest, but `pip` works too):
```bash
uv pip install -r experiments/image_retrieval/requirements.txt
```

Run the pipeline from the repo root (defaults include a built-in palette for roadway/sidewalk/crosswalk so you don’t have to pass hex codes):
```bash
python experiments/image_retrieval/pipeline.py \
  --folder /absolute/path/to/images \
  --mask-dir /absolute/path/to/masks \
  --mask-class roadway --mask-class sidewalk --mask-class crosswalk \
  --mask-focus-weight 0.6 \
  --mask-color-tolerance 6 \
  --db-name image_retrieval \
  --db-user postgres \
  --db-password postgres \
  --db-host localhost \
  --db-port 5432 \
  --out-dir experiments/image_retrieval/artifacts_clip \
  --reindex \
  --reset-table-on-dim-mismatch    # drop/recreate tables if vector dim changed
```

Custom masks/palettes
---------------------
- If you don’t pass `--mask-class`, the defaults are:
  - roadway: `(0, 128, 0)` (green)
  - sidewalk: `(0, 0, 255)` (blue)
  - crosswalk: `(255, 0, 0)` (red)
- You can override or add classes with `--mask-class`:
  - RGB or hex: `--mask-class "#00ff00:bike_lane"` or `--mask-class "0,255,0:bike_lane"`
  - Grayscale ID: `--mask-class "7:median"`
  - Comma-separated labels in one flag are supported and use defaults when known: `--mask-class "roadway,sidewalk,crosswalk"`.
  - Multiple flags also work: `--mask-class roadway --mask-class sidewalk --mask-class crosswalk`.
  - If you pass only labels (no color), known labels use the defaults above; others fall back to grayscale IDs if provided.
- If you change to a new encoder with a different dimension, either pass a new `--db-name` or add `--reset-table-on-dim-mismatch` to drop/recreate `image_embeddings` and avoid dimension errors.

Swap in BLIP embeddings by changing the encoder (defaults now isolate DB/out folders per encoder if you leave defaults untouched):
```bash
python experiments/image_retrieval/pipeline.py \
  --encoder blip \
  --blip-model Salesforce/blip-itm-base-coco \
  --folder /absolute/path/to/images \
  --db-name image_retrieval_blip \
  --out-dir experiments/image_retrieval/artifacts_blip
```

What the pipeline does:
1. Creates the target database (if needed) and prepares pgvector schema/indexes.
2. Embeds every image with OpenCLIP (batching + optional `--device cpu|mps|cuda`).
3. Upserts `(location_id, location_key, year, image_path, embedding[, mask_embedding, fusion_embedding, mask_image_embedding])` into PostgreSQL.
4. When `--mask-dir` is provided, computes segmentation-aware embeddings (class-specific deltas, fused vectors, and raw RGB mask tiles) and stores coverage stats as JSON alongside the base vectors.
5. Writes state metadata and embeddings—plus mask/fusion/mask-image matrices—to `artifacts/` (NumPy + Parquet).
6. Builds FAISS indexes for state vectors (`state_*.faiss`), mask-aware spaces (`mask_*`, `mask_image_*`), and change vectors across every available feature space (`delta_*`).
7. Computes whitening stats (`whiten.npz`) for similarity reranking.

> Tip: run `python scripts/cache_openclip.py` once to populate `experiments/image_retrieval/artifacts/model_cache` before disabling downloads (`OPENCLIP_AUTO_DOWNLOAD=0`).

Encoders (CLIP & BLIP)
----------------------
- `--encoder clip` (default) uses OpenCLIP; tune with `--clip-model` and `--clip-pretrained`.
- `--encoder blip` uses BLIP (`--blip-model Salesforce/blip-itm-base-coco` by default). The pipeline infers the embedding dimension from the model; override with `--vector-dim` only when you know you need a custom value.
- Run each encoder into its own database name and artifact folder. The CLI now always appends `_<encoder>` to whatever you pass for `--db-name` and `--out-dir` (skipping only if you already included the suffix), so FAISS/pgvector stay consistent across custom names too.

Pipeline DAG
------------
```
images → embed (CLIP/BLIP) → optional masks → (base + fusion + mask embeddings)
      ↘ cache artifacts (.npy/.parquet) ↘ upsert to Postgres/pgvector ↘ build FAISS (state, mask, Δ, fusion) ↘ whitening stats
```
Upstream inputs: image folder with year segments; optional mask folder with matching layout. Downstream surfaces: CLI queries, Streamlit dashboard, or external consumers that read the artifact bundle.

Key CLI flags:
- `--delta-consecutive-only` limits Δ vectors to adjacent years.
- `--skip-whitening` omits the whitening pass.
- `--whiten-max-samples 0` forces whitening to use the full population.
- `--reindex` refreshes the pgvector IVFFlat index after inserts.
- `--mask-dir` ingests segmentation masks (grayscale IDs or RGB tiles). Add `--mask-class <token>:<label>` per class, where `<token>` is either an integer id (`2:sidewalk`) or a colour (`#00ff00:bike_lane` / `0,120,0:median`). Use `--mask-focus-weight` to tune blending and `--mask-color-tolerance` (0–255) to allow for noisy RGB masks.
- Set `OPENCLIP_AUTO_DOWNLOAD=0` once the weights are cached to skip future network calls; the loader prints which files are missing if additional downloads are required.
- `--reuse-embeddings` skips re-embedding the base imagery so you can layer masks onto an existing run.

Artifacts layout
----------------
After a successful run `artifacts/` contains:
- `meta.parquet`, `state_rel_paths.npy`, `state_loc_keys.npy`, `state_loc_ids.npy`, `state_years.npy`
- `state_embeddings.npy`, `state_paths.npy`, plus FAISS index files for image similarity
- `fusion_embeddings.npy`, `mask_embeddings.npy`, `mask_valid.npy`, `mask_paths.npy`, `mask_stats.npy`
- `mask_image_embeddings.npy`, `mask_image_valid.npy` (CLIP vectors for the RGB mask tiles)
- `delta_embeddings.npy`, `delta_ids.npy` and FAISS index files (`delta_*.faiss`) for base image Δ vectors
- `delta_fusion_embeddings.npy`, `delta_fusion_ids.npy` and FAISS index files (`delta_fusion_*.faiss`) for mask-enhanced Δ vectors
- `delta_mask_embeddings.npy`, `delta_mask_ids.npy` and FAISS index files (`delta_mask_*.faiss`) when both years have mask coverage
- `delta_mask_image_embeddings.npy`, `delta_mask_image_ids.npy` and FAISS index files (`delta_mask_image_*.faiss`) for raw mask Δ comparisons
- `whiten.npz` (mean + whitening matrix)
`mask_stats.npy` stores per-class coverage fractions (including `background`) as JSON per image. These files drive the Streamlit app and CLI reranking. Keep the directory portable and out of git (see `.gitignore`).

`location_key` is the immutable string identifier for a physical site (path without the year segment and without the extension). `state_loc_keys.npy` stores these strings, while `state_loc_ids.npy` stores the hashed `location_id` (a deterministic 63-bit integer) for downstream systems that prefer numeric keys.

*Why whitening?* We compute a PCA whitening transform `(mu, W)` so that `z' = (z - mu) W` has unit covariance. This removes dominant directions in the CLIP space, making cosine distances more discriminative and stabilizing the FAISS rerank stage.

Sharing artifacts with collaborators
------------------------------------
Keep artifacts encoder-scoped so dimensions and indexes stay in sync:
```
experiments/image_retrieval/
  artifacts_clip/    # default when --encoder clip and --out-dir untouched
    meta.parquet, state_embeddings.npy, state_paths.npy, fusion_embeddings.npy, delta_*.faiss, whiten.npz, ...
  artifacts_blip/    # default when --encoder blip and --out-dir untouched
    meta.parquet, state_embeddings.npy, state_paths.npy, delta_*.faiss, whiten.npz, ...
  manifests/
    clip.json    # {"encoder":"clip","vector_dim":512,"db":"image_retrieval","created":"2024-..."}
    blip.json    # {"encoder":"blip","vector_dim":768,"db":"image_retrieval_blip","created":"2024-..."}
```
- Zip the encoder folder + manifest when sharing; recipients can drop it into their repo and point `query.py --artifacts <path>`/Streamlit at the bundle.
- Record the encoder name, model id, vector dimension, and DB connection string in the manifest so others can rebuild or query the exact run.
- If you share pgvector snapshots, mirror the same encoder names and dimensions (e.g., `image_retrieval` for CLIP, `image_retrieval_blip` for BLIP) to avoid cross-encoder schema clashes.

Command-line queries
--------------------
Invoke `query.py` from the repo root:
```bash
python experiments/image_retrieval/query.py \
  --mode image \
  --query /path/to/query.png \
  --top-k 8

# Search using mask-enhanced fusion vectors (requires masks ingested via --mask-dir)
python experiments/image_retrieval/query.py \
  --mode image \
  --query /path/to/query.png \
  --feature-space fusion \
  --top-k 8

# Mask-only or raw mask-image similarity
python experiments/image_retrieval/query.py \
  --mode image \
  --query /path/to/query.png \
  --feature-space mask-image \
  --top-k 8
```

Available modes:
- `image`: external image is embedded on the fly (`--dissimilar` flips to farthest matches).
- `text`: the selected encoder’s text tower (CLIP/BLIP) drives the search (same flags as image mode).
- `location`: reuse a stored embedding by `--location-key` (relative path without `/YEAR/`, no extension) and optional `--year`.
- `delta`: ranks change magnitudes for a stored location; add `--all-pairs` to compare every year combination.

All modes accept `--db-name`, `--db-host`, etc., plus `--encoder {clip,blip}` (defaults to CLIP) and an optional `--vector-dim` override (auto from the encoder when omitted). `--clip-model/--clip-pretrained` and `--blip-model` mirror the pipeline flags. Image and location modes also expose `--feature-space {image|fusion|mask|mask-image}` to pick between base embeddings, mask-enhanced fusion, mask-only deltas, or raw mask-tile embeddings.

FAISS + pgvector vs pgvector alone: pgvector gives us transactional durability, filtering (`WHERE year=2012`), and reasonable IVFFlat search in SQL. FAISS holds the same vectors in memory with HNSW/Flat indexes plus whitening-based rerank, delivering exact top-K in sub-milliseconds even when the database has millions of rows. Combined, we get the best of both worlds—Postgres for persistence & metadata joins, FAISS for high-recall similarity—rather than forcing Postgres to act as a vector engine.

Streamlit dashboard
-------------------
Launch from the repo root:
```bash
streamlit run check-similiarity/app.py \
  -- --db-name image_retrieval \
     --image-dir /absolute/path/to/images \
     --artifacts experiments/image_retrieval/artifacts
```

The optional `--artifacts` flag lets you point the Streamlit app at any precomputed cache directory.

Shared widgets:
- Sidebar slider for minimum similarity (applies to similarity and change searches).
- `Hide matches from same location` toggle removes same-location hits for similarity modes.
- Optional checkbox to show full filesystem paths alongside IDs and years.
- Feature-space selector (image / mask-enhanced fusion / mask-only / mask image) when mask artifacts are present.
- Result panels explain when fewer than `top_k` items survive the similarity cut.
- Change modes add a Δ feature selector (base, fusion, mask, mask-image) whenever the corresponding embeddings were materialised during the pipeline run.

Query modes at a glance:

| UI panel / CLI | What it does | Minimal example |
| --- | --- | --- |
| Image – Most Similar (`query.py --mode image`) | Embed an external or stored image and surface the closest matches (supports mask/fusion spaces when available). | `python experiments/image_retrieval/query.py --mode image --query /path/to/2006/4831.png --top-k 8` |
| Image – Most Dissimilar (same location) | Compare all stored years for one location to expose the largest change. | Select “Image – Most Dissimilar” in Streamlit, choose a library image; CLI: `python experiments/image_retrieval/query.py --mode location --location-key 4831 --dissimilar` |
| Text – Most Similar (`query.py --mode text`) | Use an OpenCLIP text prompt to retrieve matching imagery. | `python experiments/image_retrieval/query.py --mode text --query "protected intersection with refuge islands" --top-k 6` |
| Text → Change Δ | Describe an intervention; rank cached change vectors against the prompt (supports Δ feature spaces). | Streamlit “Text → Change Δ”, e.g. prompt `adding a protected bike lane` |
| Change Δ – Across All Years (`query.py --mode delta`) | Inspect change magnitudes for a stored location and find similar interventions elsewhere. | `python experiments/image_retrieval/query.py --mode delta --location-key 4831 --all-pairs` (UI supports mask-aware Δ spaces) |

### Image similarity
- Query via library selection or upload.
- Uses FAISS + whitening rerank when `whiten.npz` is present.
- Supports the hide-same-location toggle and minimum similarity threshold.
- Mask-aware searches display paired mask thumbnails and per-class coverage percentages with colour legends.

### Image dissimilarity (same location)
- Pick a stored image (no uploads).
- Returns the most different years for that location only (lowest cosine).
- Ideal for spotting drastic interventions within a site.

### Text → Image
- Embed natural language descriptions to find matching imagery.
- Obeys the similarity threshold and hide-same-location toggle.

### Text → Change Δ
- Describe an intervention (for example, "adding a protected bike lane").
- Text embeddings are matched against cached Δ vectors.
- Results list location id, from/to years, similarity, and optional paths.

### Change Δ explorer
- Compare two stored images via year + image id dropdowns (type-to-filter enabled).
- Defaults the second dropdown to the same image id and latest year; users can override.
- Returns the closest change signatures across all locations and year spans.
- Includes a toggle to hide results from the same location and an optional path display.

Maintenance
-----------
- Rebuild the pgvector IVFFlat index after large ingests:
  ```bash
  python - <<'PY'
  from experiments.image_retrieval.vector_db import VectorDB
  with VectorDB(dbname="image_retrieval") as db:
      db.rebuild_ivf_index(ivf_lists=200)
      db.analyze()
  PY
  ```
- Refresh whitening when the embedding distribution shifts:
  ```bash
  python experiments/image_retrieval/compute_whiten_stats.py \
    --db-name image_retrieval \
    --sample 75000 \
    --out experiments/image_retrieval/artifacts/whiten.npz
  ```
- Benchmark query latency (optional sanity check):
  ```bash
  python experiments/image_retrieval/bench_vector_db.py
  ```

With this workflow you can reproduce embeddings, keep Postgres in sync, and drive fast
similarity or change retrieval via FAISS-backed Streamlit interactions.
