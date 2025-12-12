from __future__ import annotations

import os
import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple
from functools import lru_cache

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "experiments" / "image_retrieval"


MASK_COLOR_BASE = {
    "roadway": "#008000",   # [0,128,0]
    "sidewalk": "#0000ff",  # [0,0,255]
    "crosswalk": "#ff0000", # [255,0,0]
}
MASK_COLOR_SEQUENCE = [
    "#ffb300",
    "#26a69a",
    "#7e57c2",
    "#8d6e63",
    "#29b6f6",
]
_DYNAMIC_MASK_COLORS: Dict[str, str] = {}


def _mask_color_for_label(label: str) -> str:
    key = label.lower()
    if key in MASK_COLOR_BASE:
        return MASK_COLOR_BASE[key]
    if key not in _DYNAMIC_MASK_COLORS:
        color = MASK_COLOR_SEQUENCE[len(_DYNAMIC_MASK_COLORS) % len(MASK_COLOR_SEQUENCE)]
        _DYNAMIC_MASK_COLORS[key] = color
    return _DYNAMIC_MASK_COLORS[key]


FEATURE_SPACE_OPTIONS = {
    "Image embedding": "embedding",
    "Mask-enhanced fusion": "fusion_embedding",
    "Mask-only": "mask_embedding",
    "Mask image": "mask_image_embedding",
}

DELTA_FEATURE_OPTIONS = {
    "Image Δ (base embedding)": "embedding",
    "Fusion Δ (mask-enhanced)": "fusion_embedding",
    "Mask-only Δ": "mask_embedding",
    "Mask-image Δ": "mask_image_embedding",
}


# def _load_module(module_name: str, aliases: Sequence[str]) -> ModuleType:
#     """Load module_name.py and register it under the provided aliases in sys.modules."""
#     module_path = MODULE_ROOT / f"{module_name}.py"
#     if not module_path.exists():
#         raise ModuleNotFoundError(f"Cannot locate {module_path}")
#     primary_name = aliases[0]
#     spec = importlib.util.spec_from_file_location(primary_name, module_path)
#     if spec is None or spec.loader is None:  # pragma: no cover
#         raise ImportError(f"Failed to load spec for {module_name}")
#     module = importlib.util.module_from_spec(spec)
#     for alias in aliases:
#         sys.modules[alias] = module
#     spec.loader.exec_module(module)
#     return module


# _whitening = _load_module(
#     "whitening",
#     aliases=("experiments.image_retrieval.whitening",),
# )
# _vector_db_mod = _load_module(
#     "vector_db",
#     aliases=("experiments.image_retrieval.vector_db",),
# )
# _clip_embeddings = _load_module(
#     "clip_embeddings",
#     aliases=("clip_embeddings", "experiments.image_retrieval.clip_embeddings"),
# )
# _blip_embeddings = _load_module(
#     "blip_embeddings",
#     aliases=("blip_embeddings", "experiments.image_retrieval.blip_embeddings"),
# )
# _siglip_embeddings = _load_module(
#     "siglip_embeddings",
#     aliases=("siglip_embeddings", "experiments.image_retrieval.siglip_embeddings"),
# )
# _query_rerank = _load_module(
#     "query_rerank",
#     aliases=("experiments.image_retrieval.query_rerank",),
# )

# rerank_exact_whitened = _query_rerank.rerank_exact_whitened
# load_whitening = _query_rerank.load_whitening
# CLIPEmbedder = _clip_embeddings.CLIPEmbedder
# BLIPEmbedder = _blip_embeddings.BLIPEmbedder
# SiglipEmbedder = _siglip_embeddings.SiglipEmbedder
# VectorDB = _vector_db_mod.VectorDB

import streettransformer.image_retrieval.whitening
from streettransformer.image_retrieval.vector_db import VectorDB
from streettransformer.image_retrieval.query_rerank import load_whitening
from streettransformer.image_retrieval.clip_embeddings import CLIPEmbedder
# from streettransformer.image_retrieval.blip_embeddings import BLIPEmbedder
# from streettransformer.image_retrieval.siglip_embeddings import SiglipEmbedder



# -----------------------------------------------------------------------------
# CLI arguments (supports: streamlit run app.py -- --db-name my_db)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--db-name", type=str, default="image_retrieval")
parser.add_argument("--db-user", type=str, default=os.getenv("PGUSER", "postgres"))
parser.add_argument("--db-password", type=str, default=os.getenv("PGPASSWORD", ""))
parser.add_argument("--db-host", type=str, default=os.getenv("PGHOST", "localhost"))
parser.add_argument("--db-port", type=int, default=int(os.getenv("PGPORT", "5432")))
parser.add_argument("--encoder", choices=["clip", "blip", "siglip", "eva_clip"], default="clip")
parser.add_argument("--clip-model", type=str, default="ViT-B-32")
parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
parser.add_argument("--blip-model", type=str, default="Salesforce/blip-itm-base-coco")
parser.add_argument("--siglip-model", type=str, default="google/siglip-large-patch16-384")
parser.add_argument("--eva-model", type=str, default="EVA02-CLIP-g-14")
parser.add_argument("--eva-pretrained", type=str, default="laion2b_s4b_b79k")
parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
parser.add_argument("--vector-dim", type=int, default=None,
                    help="Override embedding dimension; if omitted we infer from artifacts or encoder.")

parser.add_argument(
    "--image-dir",
    type=str,
    default=str(
        REPO_ROOT
        / "data"
        / "runtime"
        / "universes"
        / "neurips"
        / "imagery"
        / "2014"
    ),
)
parser.add_argument(
    "--artifacts",
    type=str,
    default=str(MODULE_ROOT / "artifacts"),
    help="Path to artifacts directory containing FAISS indexes, whitening, and delta caches.",
)

parsed_args, _ = parser.parse_known_args()

# Do not auto-append encoder suffixes; respect explicit names/paths passed in.
def _with_encoder_suffix(value: str) -> str:
    return value

CLI_DB_NAME = _with_encoder_suffix(parsed_args.db_name)
CLI_IMAGE_DIR = Path(parsed_args.image_dir).expanduser()
CLI_ARTIFACTS_DIR = Path(_with_encoder_suffix(parsed_args.artifacts)).expanduser()
ENCODER = parsed_args.encoder
CLIP_MODEL = parsed_args.clip_model
CLIP_PRETRAINED = parsed_args.clip_pretrained
BLIP_MODEL = parsed_args.blip_model
SIGLIP_MODEL = parsed_args.siglip_model
EVA_MODEL = parsed_args.eva_model
EVA_PRETRAINED = parsed_args.eva_pretrained
DEVICE = parsed_args.device
CLI_VECTOR_DIM = parsed_args.vector_dim


def _infer_vector_dim() -> int:
    if CLI_VECTOR_DIM:
        return int(CLI_VECTOR_DIM)
    state_path = CLI_ARTIFACTS_DIR / "state_embeddings.npy"
    if state_path.exists():
        try:
            arr = np.load(state_path, mmap_mode="r")
            if arr.ndim == 2:
                return int(arr.shape[1])
        except Exception:
            pass
    # Fallback to known defaults per encoder without forcing a heavyweight model load
    if ENCODER == "blip":
        return 768
    if ENCODER == "siglip":
        return 1024
    if ENCODER == "eva_clip":
        return 1024
    return 512


VECTOR_DIM = _infer_vector_dim()

DB_CONFIG: Dict[str, object] = {
    "dbname": CLI_DB_NAME,
    "user": parsed_args.db_user,
    "password": parsed_args.db_password,
    "host": parsed_args.db_host,
    "port": parsed_args.db_port,
    "image_dir": parsed_args.image_dir,
    "vector_dimension": VECTOR_DIM,
}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMAGE_DIR = CLI_IMAGE_DIR
# DB_CONFIG: Dict[str, object] = {
#     "dbname": CLI_DB_NAME,
#     "user": "postgres",
#     "password": "postgres",
#     "host": "localhost",
#     "port": 5432,
# }
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedder() -> CLIPEmbedder:
    # if ENCODER == "blip":
    #     return BLIPEmbedder(model_name=BLIP_MODEL, device=DEVICE)
    # if ENCODER == "siglip":
    #     return SiglipEmbedder(model_name=SIGLIP_MODEL, device=DEVICE)
    auto_flag = os.getenv("OPENCLIP_AUTO_DOWNLOAD", "1").strip().lower() not in {"0", "false", "no"}
    cache_override = os.getenv("OPENCLIP_CACHE_DIR")
    model_name = CLIP_MODEL if ENCODER != "eva_clip" else EVA_MODEL
    pretrained = CLIP_PRETRAINED if ENCODER != "eva_clip" else EVA_PRETRAINED
    return CLIPEmbedder(
        model_name=model_name,
        pretrained=pretrained,
        device=DEVICE,
        auto_download=auto_flag,
        cache_dir=cache_override,
    )


@st.cache_data(show_spinner=False)
def list_available_images(image_root: Path) -> List[Path]:
    root = Path(image_root)
    if not root.exists():
        return []
    files: List[Path] = []
    for ext in VALID_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def load_whitening_stats(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return load_whitening(path)


def _connect_db() -> VectorDB:
    db = VectorDB(**DB_CONFIG)
    db.connect()
    return db


@st.cache_data(show_spinner=False)
def fetch_metadata_for_path(path: str) -> Optional[Tuple[str, int]]:
    if path in path_metadata_map:
        return path_metadata_map[path]
    db = _connect_db()
    try:
        return db.fetch_metadata_for_path(path)
    finally:
        db.close()


def fetch_stored_entry(path: str) -> Optional[_vector_db.StoredEmbedding]:
    db = _connect_db()
    try:
        meta = db.fetch_metadata_for_path(path)
        if not meta:
            return None
        loc_id, year = meta
        rows = db.fetch_embeddings_by_location(loc_id)
        match = next((row for row in rows if row.year == year), None)
        return match
    finally:
        db.close()


@lru_cache(maxsize=8192)
def _mask_metadata_for_path(path: str) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    entry = fetch_stored_entry(path)
    if entry is None:
        return None, None
    stats = entry.mask_stats if isinstance(entry.mask_stats, dict) else None
    return entry.mask_path, stats


def _render_mask_stats(container, mask_stats: Dict[str, float], *, title: Optional[str] = None) -> None:
    if not mask_stats:
        return
    display_title = title or "Mask coverage"
    non_background = {
        cls: frac for cls, frac in mask_stats.items() if cls and cls.lower() != "background"
    }
    if non_background:
        container.markdown(f"**{display_title}:**")
        for class_name, frac in non_background.items():
            try:
                value = float(frac)
            except (TypeError, ValueError):
                continue
            color = _mask_color_for_label(class_name)
            container.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;background:{color};"
                f"margin-right:6px;'></span>{class_name}: {value * 100:.1f}%",
                unsafe_allow_html=True,
            )
    background_value = mask_stats.get("background")
    if background_value is not None:
        try:
            value = float(background_value)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            container.caption(f"Background: {value * 100:.1f}%")


def _normalize_loc_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return Path(value).stem
    except Exception:
        return value


def _format_vector_results(
    query_vec: np.ndarray,
    hits: List[_vector_db.SearchHit],
    vecs: np.ndarray,
    top_k: int,
    dissimilar: bool,
    whitelist_path: Optional[str] = "artifacts/whiten.npz",
) -> List[Dict[str, object]]:
    if not hits:
        return []

    by_path = {hit.image_path: hit for hit in hits}
    if whitelist_path:
        mu, W = load_whitening_stats(whitelist_path)
    else:
        mu = W = None
    results: List[Dict[str, object]] = []

    seen_paths: set[str] = set()
    seen_loc_year: set[Tuple[str, int]] = set()

    def _append_result(
        path: str,
        hit_obj: _vector_db.SearchHit,
        similarity: float,
        *,
        distance_override: Optional[float] = None,
    ) -> None:
        loc_norm = _normalize_loc_id(hit_obj.location_key) or hit_obj.location_key
        key_loc_year = (loc_norm, int(hit_obj.year))
        if path in seen_paths or key_loc_year in seen_loc_year:
            return
        seen_paths.add(path)
        seen_loc_year.add(key_loc_year)
        distance_val = (
            float(distance_override)
            if distance_override is not None
            else float(1.0 - similarity)
        )
        results.append(
            {
                "path": path,
                "similarity": float(similarity),
                "distance": distance_val,
                "year": hit_obj.year,
                "location_id": loc_norm,
                "location_id_raw": hit_obj.location_key,
                "mask_path": hit_obj.mask_path,
                "mask_stats": hit_obj.mask_stats,
            }
        )

    if vecs.size and mu is not None and W is not None:
        reranked = rerank_exact_whitened(
            query_vec,
            list(by_path.keys()),
            vecs,
            mu,
            W,
            top_k=len(by_path),
        )
        reranked = sorted(reranked, key=lambda item: item[1], reverse=not dissimilar)
        for path, similarity in reranked:
            hit = by_path[path]
            _append_result(path, hit, similarity)
            if len(results) >= top_k:
                break
        return results

    # Fallback: rely on raw scores from pgvector
    sorted_hits = (
        sorted(hits, key=lambda hit: hit.distance, reverse=True)
        if dissimilar
        else sorted(hits, key=lambda hit: hit.similarity, reverse=True)
    )
    for hit in sorted_hits:
        _append_result(
            hit.image_path,
            hit,
            hit.similarity,
            distance_override=hit.distance,
        )
        if len(results) >= top_k:
            break
    return results


def query_with_embedding(
    embedding: np.ndarray,
    top_k: int,
    *,
    dissimilar: bool,
    exclude_path: Optional[str] = None,
    exclude_location: Optional[str] = None,
    restrict_location: Optional[str] = None,
    min_similarity: float = 0.0,
    feature_column: str = "embedding",
) -> List[Dict[str, object]]:
    db = _connect_db()
    try:
        shortlist = max(5 * top_k, 100)
        hits = db.search_dissimilar(
            embedding,
            top_k=shortlist,
            return_metadata=True,
            exclude_path=exclude_path,
            column=feature_column,
        ) if dissimilar else db.search_similar(
            embedding,
            top_k=shortlist,
            return_metadata=True,
            exclude_path=exclude_path,
            column=feature_column,
        )
        if exclude_location:
            exclude_norm = _normalize_loc_id(exclude_location)
            hits = [
                hit
                for hit in hits
                if _normalize_loc_id(hit.location_key) != exclude_norm
            ]
        if restrict_location:
            restrict_norm = _normalize_loc_id(restrict_location)
            hits = [
                hit
                for hit in hits
                if _normalize_loc_id(hit.location_key) == restrict_norm
            ]
        dedup_hits: List[_vector_db.SearchHit] = []
        seen_paths: set[str] = set()
        for hit in hits:
            path_key = str(hit.image_path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            dedup_hits.append(hit)
        hits = dedup_hits

        ids = [hit.image_path for hit in hits]
        vecs = db.fetch_embeddings_by_paths(ids, column=feature_column)
    finally:
        db.close()
    whiten_path = ARTIFACTS_DIR / "whiten.npz" if feature_column == "embedding" else None
    results = _format_vector_results(embedding, hits, vecs, top_k, dissimilar, whitelist_path=str(whiten_path) if whiten_path and whiten_path.exists() else None)
    if not dissimilar and min_similarity > 0.0:
        results = [item for item in results if item.get("similarity", 0.0) >= min_similarity]
    return results


def run_image_query(
    image_path: Path,
    top_k: int,
    dissimilar: bool,
    *,
    exclude_location: Optional[str] = None,
    restrict_location: Optional[str] = None,
    exclude_path: Optional[str] = None,
    min_similarity: float = 0.0,
    feature_column: str = "embedding",
) -> Tuple[List[Dict[str, object]], str]:
    query_vec: Optional[np.ndarray] = None
    column = feature_column

    entry = fetch_stored_entry(str(image_path))
    if entry is not None:
        base_vec = entry.embedding.astype(np.float32, copy=False)
        if column == "fusion_embedding" and entry.fusion_embedding is not None:
            query_vec = entry.fusion_embedding.astype(np.float32, copy=False)
        elif column == "mask_embedding" and entry.mask_embedding is not None:
            query_vec = entry.mask_embedding.astype(np.float32, copy=False)
        else:
            if column != "embedding" and entry.mask_embedding is None and entry.fusion_embedding is None:
                st.warning("Mask-aware features unavailable for this image; using base embedding instead.")
            column = "embedding"
            query_vec = base_vec
    else:
        if column != "embedding":
            st.warning("Mask-aware search requires selecting a stored image with segmentation. Falling back to base embedding.")
            column = "embedding"

    if query_vec is None:
        embedder = get_embedder()
        query_vec = embedder.embed_image(str(image_path)).astype(np.float32, copy=False)
    else:
        query_vec = query_vec.astype(np.float32, copy=False)

    results = query_with_embedding(
        query_vec,
        top_k,
        dissimilar=dissimilar,
        exclude_path=exclude_path,
        exclude_location=exclude_location,
        restrict_location=restrict_location,
        min_similarity=min_similarity,
        feature_column=column,
    )
    return results, column


def run_image_dissimilar_same_location(
    image_path: Path,
    top_k: int,
) -> List[Dict[str, object]]:
    metadata = fetch_metadata_for_path(str(image_path))
    if metadata is None:
        raise ValueError("Selected image is not stored in the database. Choose from the library.")
    loc_id, year_query = metadata
    db = _connect_db()
    try:
        rows = db.fetch_embeddings_by_location(loc_id)
    finally:
        db.close()
    if not rows:
        raise ValueError(f"No embeddings available for location '{loc_id}'.")
    query_vec: Optional[np.ndarray] = None
    candidates: List[Tuple[int, np.ndarray, str, Optional[str], Optional[Dict[str, float]]]] = []
    for entry in rows:
        vec_np = entry.embedding.astype(np.float32, copy=False)
        candidates.append((int(entry.year), vec_np, str(entry.image_path), entry.mask_path, entry.mask_stats))
        if int(entry.year) == int(year_query):
            query_vec = vec_np
    if query_vec is None:
        embedder = get_embedder()
        query_vec = embedder.embed_image(str(image_path))
    query_norm = float(np.linalg.norm(query_vec)) + 1e-12
    results: List[Dict[str, object]] = []
    for yr, vec_np, path, mask_path, mask_stats in candidates:
        if int(yr) == int(year_query):
            continue
        denom = query_norm * (float(np.linalg.norm(vec_np)) + 1e-12)
        sim = float(np.dot(query_vec, vec_np) / denom)
        results.append(
            {
                "path": path,
                "similarity": sim,
                "distance": float(1.0 - sim),
                "year": yr,
                "location_id": loc_id,
                "mask_path": mask_path,
                "mask_stats": mask_stats,
            }
        )
    results.sort(key=lambda item: item["similarity"])  # lowest similarity first
    return results[:top_k]


def run_text_query(query_text: str, top_k: int, dissimilar: bool, *, min_similarity: float = 0.0) -> List[Dict[str, object]]:
    embedder = get_embedder()
    query_vec = embedder.embed_text(query_text)
    return query_with_embedding(query_vec, top_k, dissimilar=dissimilar, min_similarity=min_similarity)


def run_location_query(
    location_key: str,
    year: Optional[int],
    top_k: int,
    dissimilar: bool,
    *,
    exclude_location: Optional[str] = None,
    restrict_location: Optional[str] = None,
    min_similarity: float = 0.0,
    feature_column: str = "embedding",
) -> List[Dict[str, object]]:
    db = _connect_db()
    try:
        rows = db.fetch_embeddings_by_location(location_key)
        if not rows:
            raise ValueError(f"No embeddings found for location '{location_key}'.")
        if year is None:
            entry = rows[-1]
        else:
            matches = [row for row in rows if row.year == year]
            if not matches:
                available_years = ", ".join(str(row.year) for row in rows)
                raise ValueError(
                    f"No embedding for location '{location_key}' in {year}. "
                    f"Available years: {available_years}"
                )
            entry = matches[0]
        year_sel = entry.year
        column = feature_column
        if column == "fusion_embedding" and entry.fusion_embedding is not None:
            embedding = entry.fusion_embedding.astype(np.float32, copy=False)
        elif column == "mask_embedding" and entry.mask_embedding is not None:
            embedding = entry.mask_embedding.astype(np.float32, copy=False)
        elif column == "mask_image_embedding" and entry.mask_image_embedding is not None:
            embedding = entry.mask_image_embedding.astype(np.float32, copy=False)
        else:
            if column != "embedding":
                st.warning(
                    "Selected feature space not available for this location; using base embedding."
                )
            embedding = entry.embedding.astype(np.float32, copy=False)
            column = "embedding"
        path = entry.image_path
        embedding = embedding.astype(np.float32, copy=False)
        results = query_with_embedding(
            embedding,
            top_k,
            dissimilar=dissimilar,
            exclude_path=path,
            exclude_location=exclude_location,
            restrict_location=restrict_location,
            min_similarity=min_similarity,
            feature_column=column,
        )
    finally:
        db.close()
    return results


def run_delta_query(
    location_key: str,
    year_a: Optional[int],
    year_b: Optional[int],
    all_pairs: bool,
    top_k: int,
) -> Tuple[Optional[_vector_db.ChangeVector], List[_vector_db.ChangeVector]]:
    db = _connect_db()
    try:
        highlight = None
        if year_a is not None and year_b is not None:
            highlight = db.compare_years(location_key, year_a, year_b)
        ranked = db.rank_year_pairs(
            location_key,
            consecutive_only=not all_pairs,
        )
    finally:
        db.close()
    return highlight, ranked[:top_k]


def render_image_results(
    results: List[Dict[str, object]],
    *,
    dissimilar: bool,
    requested_top: int,
    min_similarity: float,
    show_masks: bool,
) -> None:
    if not results:
        if not dissimilar and min_similarity > 0.0:
            st.info(
                f"No results met the similarity threshold (≥ {min_similarity:.2f})."
            )
        else:
            st.info("No results returned from the vector database.")
        return
    label = "Distance" if dissimilar else "Similarity"

    if len(results) < requested_top:
        st.caption(
            f"Showing {len(results)} result(s) out of requested {requested_top}."
        )

    for item in results:
        mask_path_str = item.get("mask_path") if show_masks else None
        mask_stats = item.get("mask_stats") if show_masks else {}
        mask_stats = mask_stats or {}
        mask_path_obj: Optional[Path] = None
        mask_exists = False
        if mask_path_str:
            mask_path_obj = Path(mask_path_str)
            mask_exists = mask_path_obj.exists()

        if mask_exists:
            image_col, mask_col, info_col = st.columns([1.1, 1.1, 1.5])
        else:
            image_col, info_col = st.columns([1.2, 1.8])
            mask_col = None

        image_col.image(item["path"], caption="Image", width=320)

        if mask_exists and mask_path_obj is not None and mask_col is not None:
            mask_col.image(str(mask_path_obj), caption="Mask", width=320)

        info_col.markdown(f"**Path:** `{item['path']}`")
        if item.get("location_id"):
            info_col.markdown(f"**Location ID:** `{item['location_id']}`")
        if item.get("year") is not None:
            info_col.markdown(f"**Year:** {item['year']}")
        info_col.markdown(f"**{label}:** {item[label.lower()]:.4f}")
        if mask_exists and mask_stats:
            _render_mask_stats(info_col, mask_stats)
        st.divider()


def pick_image_ui(container, label: str, key_prefix: str) -> Tuple[Optional[Path], Optional[Path]]:
    with container:
        st.markdown(f"**{label}**")
        source_option = st.radio(
            "Image source",
            options=("Choose from library", "Upload"),
            horizontal=True,
            key=f"{key_prefix}_source",
        )

        selected: Optional[Path] = None
        temp_path: Optional[Path] = None
        if source_option == "Choose from library":
            if not available_images:
                st.warning(
                    f"No images found under `{IMAGE_DIR}`. Upload an image instead."
                )
            else:
                labels = [str(p.relative_to(IMAGE_DIR)) for p in available_images]
                choice = st.selectbox(
                    "Available images",
                    options=labels,
                    index=0,
                    key=f"{key_prefix}_library",
                )
                selected = IMAGE_DIR / Path(choice)
        else:
            uploaded = st.file_uploader(
                "Upload image",
                type=[ext.strip(".") for ext in VALID_EXTENSIONS],
                key=f"{key_prefix}_upload",
            )
            if uploaded:
                suffix = Path(uploaded.name).suffix or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp.flush()
                    temp_path = Path(tmp.name)
                selected = temp_path
        if selected:
            st.image(str(selected), caption=label, width=320)
        return selected, temp_path


def render_delta_location_results(
    highlight: Optional[_vector_db.ChangeVector],
    ranked: List[_vector_db.ChangeVector],
) -> None:
    if highlight is not None:
        st.subheader("Requested change")
        st.markdown(
            f"- Years: **{highlight.year_a} → {highlight.year_b}**  \n"
            f"- Cosine similarity: **{highlight.similarity:.4f}**  \n"
            f"- Distance: **{highlight.distance:.4f}**"
        )

    if not ranked:
        st.info("No stored change vectors for this location.")
        return

    st.subheader("Largest changes (cosine distance descending)")
    for idx, item in enumerate(ranked, start=1):
        st.markdown(
            f"{idx}. **{item.location_key}** – {item.year_a} → {item.year_b}  \n"
            f"&nbsp;&nbsp;• similarity: {item.similarity:.4f}  \n"
            f"&nbsp;&nbsp;• distance: {item.distance:.4f}"
        )


def _strip_year_segment(path: str) -> str:
    parts = Path(path).parts
    out: List[str] = []
    removed = False
    for part in parts:
        if not removed and len(part) == 4 and part.isdigit():
            removed = True
            continue
        out.append(part)
    return str(Path(*out)) if out else path


@st.cache_data(show_spinner=False)
def load_state_lookup(artifacts_dir: Path) -> Dict[Tuple[str, int], str]:
    paths = np.load(artifacts_dir / "state_paths.npy", allow_pickle=True)
    years = np.load(artifacts_dir / "state_years.npy")
    loc_key_file = artifacts_dir / "state_loc_keys.npy"
    if loc_key_file.exists():
        loc_keys = np.load(loc_key_file, allow_pickle=True)
    else:
        loc_keys = np.array([_strip_year_segment(p) for p in paths], dtype=object)

    lookup: Dict[Tuple[str, int], str] = {}
    for key, year, path in zip(loc_keys.tolist(), years.tolist(), paths.tolist()):
        lookup[(str(key), int(year))] = str(path)
    return lookup


@st.cache_data(show_spinner=False)
def load_mask_validity(artifacts_dir: Path) -> np.ndarray:
    mask_valid_file = artifacts_dir / "mask_valid.npy"
    if not mask_valid_file.exists():
        return np.array([], dtype=bool)
    return np.load(mask_valid_file)


@st.cache_data(show_spinner=False)
def load_mask_image_validity(artifacts_dir: Path) -> np.ndarray:
    mask_valid_file = artifacts_dir / "mask_image_valid.npy"
    if not mask_valid_file.exists():
        return np.array([], dtype=bool)
    return np.load(mask_valid_file)

DELTA_FEATURE_FILES = {
    "embedding": ("delta_embeddings.npy", "delta_ids.npy"),
    "fusion_embedding": ("delta_fusion_embeddings.npy", "delta_fusion_ids.npy"),
    "mask_embedding": ("delta_mask_embeddings.npy", "delta_mask_ids.npy"),
    "mask_image_embedding": ("delta_mask_image_embeddings.npy", "delta_mask_image_ids.npy"),
}


@st.cache_resource(show_spinner=False)
def load_delta_matrix(artifacts_dir: Path, feature: str = "embedding"):
    feature_key = feature.lower()
    if feature_key not in DELTA_FEATURE_FILES:
        raise ValueError(f"Unsupported Δ feature space '{feature}'.")
    emb_file, id_file = DELTA_FEATURE_FILES[feature_key]
    embeddings_path = artifacts_dir / emb_file
    ids_path = artifacts_dir / id_file
    if not embeddings_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Delta embeddings or ids for feature '{feature}' not found under {artifacts_dir}."
        )
    embeddings = np.load(embeddings_path, mmap_mode="r")
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    return embeddings, ids


@lru_cache(maxsize=4)
def available_delta_features(artifacts_dir: str) -> List[str]:
    root = Path(artifacts_dir)
    features: List[str] = []
    for key, (emb_file, id_file) in DELTA_FEATURE_FILES.items():
        if (root / emb_file).exists() and (root / id_file).exists():
            features.append(key)
    return features


def build_delta_feature_choices(artifacts_dir: Path) -> List[Tuple[str, str]]:
    available = set(available_delta_features(str(artifacts_dir)))
    choices: List[Tuple[str, str]] = []
    for label, key in DELTA_FEATURE_OPTIONS.items():
        if key in available:
            choices.append((label, key))
    return choices


def _compute_delta_vector(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    diff = vec_b.astype(np.float32) - vec_a.astype(np.float32)
    norm = np.linalg.norm(diff)
    if norm < 1e-12:
        return np.zeros_like(diff, dtype=np.float32)
    return diff / norm


def _select_feature_vector(entry: _vector_db.StoredEmbedding, feature_column: str) -> Optional[np.ndarray]:
    key = feature_column.lower()
    if key in {"fusion", "fusion_embedding"}:
        return entry.fusion_embedding
    if key in {"mask", "mask_embedding"}:
        return entry.mask_embedding
    if key in {"mask_image", "mask_image_embedding"}:
        return entry.mask_image_embedding
    return entry.embedding


def _has_feature(path: Path, feature_column: str) -> bool:
    entry = fetch_stored_entry(str(path))
    if entry is None:
        return False
    return _select_feature_vector(entry, feature_column) is not None


def _embedding_from_path(
    path: Path,
    embedder: CLIPEmbedder,
    feature_column: str = "embedding",
) -> np.ndarray:
    str_path = str(path)
    entry = fetch_stored_entry(str_path)
    if entry is not None:
        vec = _select_feature_vector(entry, feature_column)
        if vec is not None:
            return vec.astype(np.float32, copy=False)
        if feature_column.lower() != "embedding":
            raise ValueError(
                f"Stored image '{str_path}' lacks {feature_column} features. "
                "Choose an image with segmentation-derived embeddings."
            )
        return entry.embedding.astype(np.float32, copy=False)
    db = _connect_db()
    try:
        vecs = db.fetch_embeddings_by_paths([str_path], column=feature_column)
    finally:
        db.close()
    if vecs.shape[0] == 1:
        return vecs[0]
    if feature_column.lower() != "embedding":
        raise ValueError(
            "Mask-aware Δ queries require stored embeddings with segmentation. "
            "Select images from the library that were processed with masks."
        )
    return embedder.embed_image(str_path).astype(np.float32, copy=False)


def _explain_delta_id(delta_id: str, lookup: Dict[Tuple[str, int], str]) -> Tuple[str, str]:
    if "::" not in delta_id or "->" not in delta_id:
        return ("<unknown>", "<unknown>")
    prefix, yrs = delta_id.rsplit("::", 1)
    try:
        year_a_str, year_b_str = yrs.split("->")
        year_a = int(year_a_str)
        year_b = int(year_b_str)
    except ValueError:
        return ("<unknown>", "<unknown>")
    from_path = lookup.get((prefix, year_a), "<unknown>")
    to_path = lookup.get((prefix, year_b), "<unknown>")
    return (from_path, to_path)


def _parse_delta_id(delta_id: str) -> Tuple[str, Optional[int], Optional[int]]:
    if "::" not in delta_id or "->" not in delta_id:
        return delta_id, None, None
    prefix, yrs = delta_id.rsplit("::", 1)
    try:
        year_a_str, year_b_str = yrs.split("->")
        return prefix, int(year_a_str), int(year_b_str)
    except ValueError:
        return prefix, None, None


def _query_delta_neighbors(
    delta_vec: np.ndarray,
    top_k: int,
    artifacts_dir: Path,
    min_similarity: float,
    feature_column: str = "embedding",
) -> List[Dict[str, object]]:
    embeddings, delta_ids = load_delta_matrix(artifacts_dir, feature_column)
    if embeddings.shape[0] == 0:
        return []
    lookup = load_state_lookup(artifacts_dir)

    vec = delta_vec.astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return []
    vec /= norm

    # embeddings are already normalised when created; use cosine similarity
    scores = embeddings @ vec
    order = np.argsort(scores)[::-1]

    neighbors: List[Dict[str, object]] = []
    for idx in order:
        delta_id = str(delta_ids[idx])
        sim = float(scores[idx])
        from_path, to_path = _explain_delta_id(delta_id, lookup)
        loc_id, year_a, year_b = _parse_delta_id(delta_id)
        delta_loc = loc_id or _strip_year_segment(from_path)
        loc_key_norm = _normalize_loc_id(delta_loc)
        if sim < min_similarity:
            continue
        mask_from_path = mask_from_stats = mask_to_path = mask_to_stats = None
        if from_path not in {"<unknown>", ""}:
            mask_from_path, mask_from_stats = _mask_metadata_for_path(str(from_path))
        if to_path not in {"<unknown>", ""}:
            mask_to_path, mask_to_stats = _mask_metadata_for_path(str(to_path))
        neighbors.append(
            {
                "delta_id": delta_id,
                "similarity": sim,
                "distance": float(1.0 - sim),
                "from_path": from_path,
                "to_path": to_path,
                "location_id": delta_loc,
                "location_norm": loc_key_norm,
                "year_a": year_a,
                "year_b": year_b,
                "mask_from_path": mask_from_path,
                "mask_from_stats": mask_from_stats,
                "mask_to_path": mask_to_path,
                "mask_to_stats": mask_to_stats,
            }
        )
        if len(neighbors) >= top_k:
            break
    return neighbors


def run_delta_query_from_images(
    image_a: Path,
    image_b: Path,
    top_k: int,
    *,
    artifacts_dir: Path,
    min_similarity: float,
    feature_column: str,
    exclude_location: Optional[str] = None,
) -> List[Dict[str, object]]:
    embedder = get_embedder()
    vec_a = _embedding_from_path(image_a, embedder, feature_column)
    vec_b = _embedding_from_path(image_b, embedder, feature_column)
    delta_vec = _compute_delta_vector(vec_a, vec_b)
    neighbors = _query_delta_neighbors(
        delta_vec,
        top_k,
        artifacts_dir,
        min_similarity,
        feature_column=feature_column,
    )
    if exclude_location:
        exclude_norm = _normalize_loc_id(exclude_location)
        neighbors = [
            item
            for item in neighbors
            if item.get("location_norm") != exclude_norm
        ]
    return neighbors


def run_delta_query_from_text(
    description: str,
    top_k: int,
    *,
    artifacts_dir: Path,
    min_similarity: float,
    feature_column: str,
) -> List[Dict[str, object]]:
    embedder = get_embedder()
    text_vec = embedder.embed_text(description).astype(np.float32, copy=False)
    # Ensure vector dim matches the delta matrix; resize/abort on mismatch
    if text_vec.shape[0] != VECTOR_DIM:
        st.error(
            f"Text embedding dim ({text_vec.shape[0]}) does not match loaded vectors ({VECTOR_DIM}). "
            "Check that the Streamlit encoder matches the artifacts (e.g., --encoder blip for BLIP artifacts)."
        )
        return []
    return _query_delta_neighbors(
        text_vec,
        top_k,
        artifacts_dir,
        min_similarity,
        feature_column=feature_column,
    )


def render_delta_neighbors(
    base_a: Path,
    base_b: Path,
    neighbors: List[Dict[str, object]],
    *,
    exclude_same_location: bool,
    show_paths: bool,
    base_location_id: Optional[str],
    min_similarity: float,
    requested_top: int,
    feature_label: Optional[str] = None,
    show_masks: bool = False,
) -> None:
    st.subheader("Reference change")
    st.markdown(f"**Δ:** `{base_a}` → `{base_b}`")
    if feature_label:
        st.caption(f"Feature space: {feature_label}")
    if show_masks:
        mask_base_a_path, mask_base_a_stats = _mask_metadata_for_path(str(base_a))
        mask_base_b_path, mask_base_b_stats = _mask_metadata_for_path(str(base_b))
    else:
        mask_base_a_path = mask_base_b_path = None
        mask_base_a_stats = mask_base_b_stats = None

    base_cols = st.columns(2)
    base_a_col, base_b_col = base_cols
    if base_a.exists():
        base_a_col.image(str(base_a), caption="Baseline (from)", width=260)
    if show_masks and mask_base_a_path:
        mask_path_a = Path(mask_base_a_path)
        if mask_path_a.exists():
            base_a_col.image(str(mask_path_a), caption="Mask", width=260)
            _render_mask_stats(base_a_col, mask_base_a_stats or {})

    if base_b.exists():
        base_b_col.image(str(base_b), caption="Comparison (to)", width=260)
    if show_masks and mask_base_b_path:
        mask_path_b = Path(mask_base_b_path)
        if mask_path_b.exists():
            base_b_col.image(str(mask_path_b), caption="Mask", width=260)
            _render_mask_stats(base_b_col, mask_base_b_stats or {})

    if not neighbors:
        if min_similarity > 0.0:
            st.info(
                f"No Δ vectors passed the similarity threshold (≥ {min_similarity:.2f})."
            )
        else:
            st.info("No similar changes found in the delta index.")
        return
    st.subheader("Similar changes across the dataset")
    base_prefix = base_location_id
    if base_prefix is None:
        base_meta = fetch_metadata_for_path(str(base_a))
        if base_meta:
            base_prefix = base_meta[0]

    filtered: List[Dict[str, object]] = []
    display_idx = 0
    for item in neighbors:
        loc_id_raw = item.get("location_id") or _strip_year_segment(item["from_path"])
        loc_norm = item.get("location_norm") or _normalize_loc_id(loc_id_raw)
        if exclude_same_location and base_prefix:
            if loc_norm == _normalize_loc_id(base_prefix):
                continue
        filtered.append(item)

    if not filtered:
        st.info("No results after applying location filter.")
        return

    if len(filtered) < requested_top:
        st.caption(f"Showing {len(filtered)} result(s) out of requested {requested_top}.")

    for item in filtered[:requested_top]:
        display_idx += 1
        col_from, col_to, col_info = st.columns([1.1, 1.1, 1.6])
        from_path = Path(item["from_path"]).expanduser()
        to_path = Path(item["to_path"]).expanduser()
        with col_from:
            if from_path.exists():
                st.image(str(from_path), caption="From", width=260)
            else:  # pragma: no cover - missing artifact
                st.warning(f"Missing image: {from_path}")
            if show_masks:
                mask_from_path = item.get("mask_from_path")
                if mask_from_path:
                    mask_from_obj = Path(mask_from_path)
                    if mask_from_obj.exists():
                        col_from.image(str(mask_from_obj), caption="Mask", width=240)
                        _render_mask_stats(col_from, item.get("mask_from_stats") or {})
        with col_to:
            if to_path.exists():
                st.image(str(to_path), caption="To", width=260)
            else:  # pragma: no cover
                st.warning(f"Missing image: {to_path}")
            if show_masks:
                mask_to_path = item.get("mask_to_path")
                if mask_to_path:
                    mask_to_obj = Path(mask_to_path)
                    if mask_to_obj.exists():
                        col_to.image(str(mask_to_obj), caption="Mask", width=240)
                        _render_mask_stats(col_to, item.get("mask_to_stats") or {})
        with col_info:
            st.markdown(
                f"**{display_idx}. Δ id:** `{item['delta_id']}`  \n"
                f"• location ID: `{item.get('location_id')}`  \n"
                f"• years: {item.get('year_a')} → {item.get('year_b')}  \n"
                f"• similarity: {item['similarity']:.4f}  \n"
                f"• distance: {item['distance']:.4f}"
            )
            if show_paths:
                st.markdown(f"• from path: `{from_path}`  \n• to path: `{to_path}`")
        st.divider()


def render_delta_text_results(
    description: str,
    neighbors: List[Dict[str, object]],
    *,
    show_paths: bool,
    min_similarity: float,
    requested_top: int,
    feature_label: Optional[str] = None,
    show_masks: bool = False,
) -> None:
    st.subheader("Query description")
    st.markdown(f"`{description}`")
    if feature_label:
        st.caption(f"Feature space: {feature_label}")
    if not neighbors:
        if min_similarity > 0.0:
            st.info(
                f"No Δ vectors matched the description above the similarity threshold (≥ {min_similarity:.2f})."
            )
        else:
            st.info("No change signatures matched the description.")
        return
    if len(neighbors) < requested_top:
        st.caption(f"Showing {len(neighbors)} result(s) out of requested {requested_top}.")
    for idx, item in enumerate(neighbors[:requested_top], start=1):
        from_path = Path(item["from_path"]).expanduser()
        to_path = Path(item["to_path"]).expanduser()
        loc_id = item.get("location_id")
        col_from, col_to, col_info = st.columns([1.1, 1.1, 1.6])
        with col_from:
            if from_path.exists():
                st.image(str(from_path), caption="From", width=260)
            if show_masks:
                mask_from_path = item.get("mask_from_path")
                if mask_from_path:
                    mask_from_obj = Path(mask_from_path)
                    if mask_from_obj.exists():
                        col_from.image(str(mask_from_obj), caption="Mask", width=240)
                        _render_mask_stats(col_from, item.get("mask_from_stats") or {})
        with col_to:
            if to_path.exists():
                st.image(str(to_path), caption="To", width=260)
            if show_masks:
                mask_to_path = item.get("mask_to_path")
                if mask_to_path:
                    mask_to_obj = Path(mask_to_path)
                    if mask_to_obj.exists():
                        col_to.image(str(mask_to_obj), caption="Mask", width=240)
                        _render_mask_stats(col_to, item.get("mask_to_stats") or {})
        with col_info:
            st.markdown(
                f"**{idx}. Δ id:** `{item['delta_id']}`  \n"
                f"• location ID: `{loc_id}`  \n"
                f"• years: {item.get('year_a')} → {item.get('year_b')}  \n"
                f"• similarity: {item['similarity']:.4f}  \n"
                f"• distance: {item['distance']:.4f}"
            )
            if show_paths:
                st.markdown(f"• from path: `{from_path}`  \n• to path: `{to_path}`")
        st.divider()


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Change Similarity Explorer",
    page_icon="🛰️",
    layout="wide",
)

st.title("Change Similarity Explorer")
st.caption(
    "Explore similar or contrasting streetscape imagery and surface change signatures across years."
)

available_images = list_available_images(IMAGE_DIR)
ARTIFACTS_DIR = CLI_ARTIFACTS_DIR
mask_valid_global = load_mask_validity(ARTIFACTS_DIR)
mask_image_valid_global = load_mask_image_validity(ARTIFACTS_DIR)
MASK_SUPPORT = bool(mask_valid_global.size and mask_valid_global.any()) or bool(
    mask_image_valid_global.size and mask_image_valid_global.any()
)
DELTA_FEATURE_CHOICES = build_delta_feature_choices(ARTIFACTS_DIR)
DEFAULT_DELTA_FEATURE = "fusion_embedding" if any(
    choice[1] == "fusion_embedding" for choice in DELTA_FEATURE_CHOICES
) else "embedding"


def _extract_year_from_relative(rel: str) -> Optional[int]:
    for part in Path(rel).parts:
        if len(part) == 4 and part.isdigit():
            return int(part)
    return None


def build_library_index(images: List[Path]) -> Dict[str, Dict[str, Optional[object]]]:
    info: Dict[str, Dict[str, Optional[object]]] = {}
    for path in images:
        try:
            rel = str(path.relative_to(IMAGE_DIR))
        except ValueError:
            rel = str(path)
        loc_id = _strip_year_segment(rel)
        year = _extract_year_from_relative(rel)
        info[rel] = {"loc_id": loc_id, "year": year}
    return info


library_index = build_library_index(available_images)


@st.cache_data(show_spinner=False)
def load_path_metadata() -> Dict[str, Tuple[str, int]]:
    artifacts_dir = ARTIFACTS_DIR
    paths_file = artifacts_dir / "state_paths.npy"
    years_file = artifacts_dir / "state_years.npy"
    if not paths_file.exists() or not years_file.exists():
        return {}
    paths = np.load(paths_file, allow_pickle=True)
    years = np.load(years_file)
    loc_keys_file = artifacts_dir / "state_loc_keys.npy"
    if loc_keys_file.exists():
        loc_keys = np.load(loc_keys_file, allow_pickle=True)
    else:
        loc_keys = np.array([_strip_year_segment(str(p)) for p in paths], dtype=object)
    meta: Dict[str, Tuple[str, int]] = {}
    for p, loc_id, year in zip(paths.tolist(), loc_keys.tolist(), years.tolist()):
        meta[str(p)] = (str(loc_id), int(year))
    return meta


def build_location_catalog(meta: Dict[str, Tuple[str, int]]) -> Dict[str, Dict[int, Path]]:
    catalog: Dict[str, Dict[int, Path]] = {}
    for path_str, (loc_id, year) in meta.items():
        path_obj = Path(path_str)
        catalog.setdefault(loc_id, {})[year] = path_obj
    return {loc: years for loc, years in catalog.items() if years}


path_metadata_map = load_path_metadata()
location_catalog = build_location_catalog(path_metadata_map)

QUERY_OPTIONS = [
    ("Image – Most Similar", "image_sim", False),
    ("Image – Most Dissimilar", "image_dis", True),
    ("Text – Most Similar", "text_sim", False),
    ("Text → Change Δ", "delta_text", False),
    ("Change Δ – Across All Years", "delta", False),
]

with st.sidebar:
    st.header("Query Settings")
    query_label = st.selectbox("Query Type", [opt[0] for opt in QUERY_OPTIONS], index=0)
    query_key, is_dissimilar = next(
        (key, dis) for label, key, dis in QUERY_OPTIONS if label == query_label
    )
    top_k = st.slider("Top K Results", min_value=1, max_value=30, value=10)
    min_similarity = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Floor cosine similarity for similar queries (ignored for dissimilar mode).",
    )

    st.markdown("---")
    st.subheader("Connection")
    st.markdown(f"**Database:** {DB_CONFIG['dbname']}")
    st.markdown(f"**Host:** {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    st.markdown(f"**Image root:** `{IMAGE_DIR}`")
    st.markdown(f"**Indexed images:** {len(available_images)} (library)")
    st.markdown(
        "<span style='font-size: 0.85rem;'>Run "
        "`streamlit run check-similiarity/app.py -- --db-name your_db --image-dir /path/to/images` "
        "to point at different resources.</span>",
        unsafe_allow_html=True,
    )


if query_key == "image_sim":
    st.subheader("Image query")
    source_option = st.radio(
        "Image source",
        options=("Choose from library", "Upload from device"),
        horizontal=True,
    )

    selected_image: Optional[Path] = None
    temp_path: Optional[Path] = None
    query_location: Optional[str] = None
    query_year: Optional[int] = None
    feature_column = "embedding"

    if source_option == "Choose from library":
        if not available_images:
            st.warning(
                f"No images found under `{IMAGE_DIR}`. "
                "Upload an image instead or update the image directory."
            )
        else:
            labels = [str(p.relative_to(IMAGE_DIR)) for p in available_images]
            choice = st.selectbox("Available images", options=labels, index=0)
            selected_image = IMAGE_DIR / Path(choice)
    else:
        uploaded = st.file_uploader(
            "Upload an image",
            type=[ext.strip(".") for ext in VALID_EXTENSIONS],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                temp_path = Path(tmp.name)
            selected_image = temp_path

    if selected_image:
        metadata = fetch_metadata_for_path(str(selected_image))
        if metadata is not None:
            query_location, query_year = metadata
        else:
            query_location = query_year = None
        st.image(str(selected_image), caption="Query image", width=420)
        if query_location is not None and query_year is not None:
            st.caption(f"Location ID: `{query_location}` · Year: {query_year}")

        feature_choices = ["Image embedding"]
        default_idx = 0
        if MASK_SUPPORT:
            feature_choices.extend(["Mask-enhanced fusion", "Mask-only", "Mask image"])
            default_idx = 1
        feature_option = st.selectbox(
            "Feature space",
            feature_choices,
            index=default_idx,
            help="Choose which embedding space to search. Mask-aware options require ingested segmentation masks.",
        )
        feature_column = FEATURE_SPACE_OPTIONS[feature_option]

        hide_same_loc = False
        restrict_same_loc = False
        if query_location:
            hide_same_loc = st.checkbox(
                "Hide results from this location",
                value=True,
                help="Skip matches that map to the same location id as the query image.",
            )
    else:
        query_location = None
        hide_same_loc = False
        restrict_same_loc = False

    if st.button("Find Most Similar Images", type="primary", disabled=selected_image is None):
        try:
            exclude_loc = query_location if (hide_same_loc and query_location) else None
            with st.spinner("Searching..."):
                results, used_column = run_image_query(
                    selected_image,  # type: ignore[arg-type]
                    top_k,
                    dissimilar=False,
                    exclude_location=exclude_loc,
                    restrict_location=None,
                    exclude_path=str(selected_image),
                    min_similarity=min_similarity,
                    feature_column=feature_column,
                )
            render_image_results(
                results,
                dissimilar=False,
                requested_top=top_k,
                min_similarity=min_similarity,
                show_masks=used_column != "embedding",
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Image query failed: {exc}")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

elif query_key == "image_dis":
    st.subheader("Image dissimilarity within the same location")
    source_option = st.radio(
        "Image source",
        options=("Choose from library", "Upload from device"),
        horizontal=True,
    )
    temp_path: Optional[Path] = None
    selected_image: Optional[Path] = None
    if source_option == "Choose from library":
        if not available_images:
            st.warning(
                f"No images found under `{IMAGE_DIR}`."
            )
        else:
            labels = [str(p.relative_to(IMAGE_DIR)) for p in available_images]
            choice = st.selectbox("Available images", options=labels, index=0)
            selected_image = IMAGE_DIR / Path(choice)
    else:
        uploaded = st.file_uploader(
            "Upload an image",
            type=[ext.strip(".") for ext in VALID_EXTENSIONS],
            help="Stored embeddings are required; uploads are converted on the fly but should map to an existing location.",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp.flush()
                temp_path = Path(tmp.name)
            selected_image = temp_path
            st.warning("Uploaded image may not correspond to a stored location; prefer selecting from the library.")

    if selected_image and selected_image.exists():
        st.image(str(selected_image), caption="Query image", width=420)
        meta = fetch_metadata_for_path(str(selected_image))
        if meta is not None:
            st.caption(f"Location ID: `{meta[0]}` · Year: {meta[1]}")

    if st.button("Find Most Dissimilar Images", type="primary", disabled=selected_image is None):
        try:
            with st.spinner("Searching..."):
                results = run_image_dissimilar_same_location(selected_image, top_k)  # type: ignore[arg-type]
            render_image_results(
                results,
                dissimilar=True,
                requested_top=top_k,
                min_similarity=0.0,
                show_masks=False,
            )
        except Exception as exc:
            st.error(f"Image query failed: {exc}")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

elif query_key == "text_sim":
    st.subheader("Text query")
    prompt = st.text_area(
        "Describe the scene",
        value="an aerial view of a protected intersection with refuge islands",
        help="A CLIP text embedding drives the search. Detailed prompts generally yield better matches.",
    )
    if st.button("Find Most Similar Images", type="primary", disabled=not prompt.strip()):
        try:
            with st.spinner("Searching..."):
                results = run_text_query(
                    prompt.strip(),
                    top_k,
                    dissimilar=False,
                    min_similarity=min_similarity,
                )
            render_image_results(
                results,
                dissimilar=False,
                requested_top=top_k,
                min_similarity=min_similarity,
                show_masks=False,
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Text query failed: {exc}")

elif query_key == "delta_text":
    st.subheader("Describe a change to retrieve similar Δ signatures")
    description = st.text_area(
        "Change description",
        value="adding a protected bike lane and reducing car lanes",
        help="Describe the intervention; the text is embedded with CLIP and compared against stored Δ vectors.",
    )
    delta_choices = DELTA_FEATURE_CHOICES or [("Image Δ (base embedding)", "embedding")]
    delta_mapping = {label: key for label, key in delta_choices}
    default_idx = 0
    for idx, (_, key) in enumerate(delta_choices):
        if key == DEFAULT_DELTA_FEATURE:
            default_idx = idx
            break
    selected_delta_label = st.selectbox(
        "Δ feature space",
        [label for label, _ in delta_choices],
        index=default_idx,
        help="Blend of embeddings used to compare change signatures.",
        key="delta_text_feature",
    )
    selected_delta_feature = delta_mapping[selected_delta_label]
    show_paths_toggle = st.checkbox(
        "Show full file paths",
        value=False,
        help="Include absolute paths in the results table.",
        key="delta_text_show_paths",
    )
    if st.button("Find Changes", type="primary", disabled=not description.strip()):
        try:
            with st.spinner("Searching change signatures..."):
                neighbors = run_delta_query_from_text(
                    description.strip(),
                    top_k,
                    artifacts_dir=ARTIFACTS_DIR,
                    min_similarity=min_similarity,
                    feature_column=selected_delta_feature,
                )
            render_delta_text_results(
                description.strip(),
                neighbors,
                show_paths=show_paths_toggle,
                min_similarity=min_similarity,
                requested_top=top_k,
                feature_label=selected_delta_label,
                show_masks=selected_delta_feature != "embedding",
            )
        except Exception as exc:
            st.error(f"Δ query failed: {exc}")

else:  # delta mode
    st.subheader("Change similarity (Δ vectors)")

    st.markdown("Select the **baseline** (year A) and **comparison** (year B) imagery. Use the dropdowns below or toggle to upload custom files.")

    use_uploads = st.checkbox(
        "Upload custom images",
        value=False,
        help="Toggle to upload your own before/after images instead of using the library.",
    )

    temp_a: Optional[Path] = None
    temp_b: Optional[Path] = None
    selected_loc = None
    year_a_val = year_b_val = None

    if use_uploads:
        col_left, col_right = st.columns(2)
        img_a, temp_a = pick_image_ui(col_left, "Baseline (from) image", "delta_a_upload")
        img_b, temp_b = pick_image_ui(col_right, "Comparison (to) image", "delta_b_upload")
        loc_id_a = loc_id_b = None
    else:
        if not location_catalog:
            st.warning("Library metadata is missing; switch to uploads to compare arbitrary images.")
            img_a = img_b = None
            loc_id_a = loc_id_b = None
        else:
            loc_options = sorted(location_catalog.keys())
            selected_loc = st.selectbox(
                "Location id",
                options=loc_options,
                help="Pick the location whose change you want to inspect.",
            )
            years_available = sorted(location_catalog[selected_loc].keys())
            if len(years_available) < 2:
                st.warning("Need at least two years of imagery for this location.")
            year_a_val = st.selectbox(
                "Baseline year",
                options=years_available,
                index=0,
                key="delta_year_a",
            )
            remaining_years = [y for y in years_available if y != year_a_val]
            comparison_years = remaining_years if remaining_years else years_available
            year_b_val = st.selectbox(
                "Comparison year",
                options=comparison_years,
                index=0,
                key="delta_year_b",
            )
            img_a = location_catalog[selected_loc].get(int(year_a_val))
            img_b = location_catalog[selected_loc].get(int(year_b_val))
            loc_id_a = loc_id_b = selected_loc
            col_preview_a, col_preview_b = st.columns(2)
            if img_a and img_a.exists():
                col_preview_a.image(str(img_a), caption=f"Baseline ({selected_loc}, {year_a_val})", width=320)
            if img_b and img_b.exists():
                col_preview_b.image(str(img_b), caption=f"Comparison ({selected_loc}, {year_b_val})", width=320)
    artifacts_dir = ARTIFACTS_DIR
    disabled = img_a is None or img_b is None or not artifacts_dir.exists()
    exclude_same_loc_toggle = st.checkbox(
        "Hide matches from the same location",
        value=False,
        help="Suppress deltas whose before/after paths match the query location.",
        key="delta_exclude_toggle",
    )
    show_paths_toggle = st.checkbox(
        "Show full file paths",
        value=False,
        help="Include absolute paths in the results table.",
        key="delta_show_paths",
    )
    delta_choices = DELTA_FEATURE_CHOICES or [("Image Δ (base embedding)", "embedding")]
    delta_mapping = {label: key for label, key in delta_choices}
    default_delta_idx = 0
    for idx, (_, key) in enumerate(delta_choices):
        if key == DEFAULT_DELTA_FEATURE:
            default_delta_idx = idx
            break
    selected_delta_label = st.selectbox(
        "Δ feature space",
        [label for label, _ in delta_choices],
        index=default_delta_idx,
        help="Select which embeddings to use when computing change vectors.",
        key="delta_feature_space",
    )
    selected_delta_feature = delta_mapping[selected_delta_label]
    if use_uploads and selected_delta_feature != "embedding":
        st.info(
            "Mask-aware Δ search requires stored imagery with segmentation. "
            "Select images from the library or switch the feature space to the base embedding."
        )

    base_loc = loc_id_a if loc_id_a is not None else None

    feature_missing_path: Optional[Path] = None
    if selected_delta_feature != "embedding":
        for cand in (img_a, img_b):
            if cand is not None and not _has_feature(cand, selected_delta_feature):
                feature_missing_path = cand
                break
        if feature_missing_path:
            st.error(
                f"Selected feature '{selected_delta_feature}' not available for {feature_missing_path.name}. "
                "Pick images processed with masks or switch to base embeddings."
            )
            disabled = True

    if st.button("Find Similar Changes", type="primary", disabled=disabled):
        try:
            with st.spinner("Searching change signatures..."):
                neighbors = run_delta_query_from_images(
                    img_a,  # type: ignore[arg-type]
                    img_b,  # type: ignore[arg-type]
                    top_k,
                    artifacts_dir=artifacts_dir,
                    min_similarity=min_similarity,
                    feature_column=selected_delta_feature,
                    exclude_location=base_loc,
                )
            render_delta_neighbors(
                img_a,  # type: ignore[arg-type]
                img_b,  # type: ignore[arg-type]
                neighbors,
                exclude_same_location=exclude_same_loc_toggle,
                show_paths=show_paths_toggle,
                base_location_id=base_loc,
                min_similarity=min_similarity,
                requested_top=top_k,
                feature_label=selected_delta_label,
                show_masks=selected_delta_feature != "embedding",
            )
        except Exception as exc:
            st.error(f"Δ query failed: {exc}")
        finally:
            for temp in (temp_a, temp_b):
                if temp is not None and temp.exists():
                    temp.unlink(missing_ok=True)
