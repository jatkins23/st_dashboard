import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, cast

import numpy as np

from clip_embeddings import CLIPEmbedder
from vector_db import ChangeVector, SearchHit, StoredEmbedding, VectorDB

EmbedderType = CLIPEmbedder


def _make_embedder(
    encoder: str,
    *,
    clip_model: str,
    clip_pretrained: str,
    eva_model: str,
    eva_pretrained: str,
    device: str | None = None,
) -> EmbedderType:
    name = encoder.strip().lower()
    if name == "clip":
        return CLIPEmbedder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device
            )
    if name == "eva_clip":
        return CLIPEmbedder(
            model_name=eva_model,
            pretrained=eva_pretrained,
            device=device,
        )
    raise ValueError(f"Unsupported encoder '{encoder}'. Choose from: clip, eva_clip.")


def _feature_to_column(name: str) -> str:
    mapping = {
        "image": "embedding",
        "fusion": "fusion_embedding",
        "mask": "mask_embedding",
        "mask-image": "mask_image_embedding",
        "mask_image": "mask_image_embedding",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported feature space '{name}'")
    return mapping[name]



def _print_hits(
        title: str, 
        hits: Iterable[SearchHit]
        ) -> None:
    hits = list(hits)
    if not hits:
        print(f"No results for {title}.")
        return
    print("=" * 72)
    print(title)
    print("-" * 72)
    for idx, hit in enumerate(hits, start=1):
        print(f"{idx:2d}. {hit.image_path}")
        print(f"     location={hit.location_key}  year={hit.year}  similarity={hit.similarity:.4f}  distance={hit.distance:.4f}")
    print()


def _print_changes(
        title: str, 
        rows: Iterable[ChangeVector], 
        limit: int | None = None
        ) -> None:
    rows = list(rows)
    if not rows:
        print(f"No change vectors available for {title}.")
        return
    print("=" * 72)
    print(title)
    print("-" * 72)
    for idx, item in enumerate(rows, start=1):
        if limit is not None and idx > limit:
            break
        print(
            f"{idx:2d}. {item.location_key} {item.year_a}->{item.year_b}  "
            f"similarity={item.similarity:.4f}  distance={item.distance:.4f}"
        )
    print()



def query_by_image(
    image_path: str,
    db_cfg: Dict[str, object],
    *,
    top_k: int,
    dissimilar: bool,
    vector_dim: int,
    column: str,
    embedder: EmbedderType,
) -> List[SearchHit]:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
        metadata = db.fetch_metadata_for_path(image_path)
        entry: Optional[StoredEmbedding] = None
        if metadata is not None:
            loc_id, year = metadata
            rows = db.fetch_embeddings_by_location(loc_id)
            entry = next((row for row in rows if row.year == year), None)

        selected_column = column
        query_vec: Optional[np.ndarray] = None

        if entry is not None:
            base_vec = entry.embedding.astype(np.float32, copy=False)
            if column == "fusion_embedding" and entry.fusion_embedding is not None:
                query_vec = entry.fusion_embedding.astype(np.float32, copy=False)
            elif column == "mask_embedding" and entry.mask_embedding is not None:
                query_vec = entry.mask_embedding.astype(np.float32, copy=False)
            elif column == "mask_image_embedding" and entry.mask_image_embedding is not None:
                query_vec = entry.mask_image_embedding.astype(np.float32, copy=False)
            else:
                if column != "embedding":
                    print("[info] Requested feature space missing for this image; using base embedding.")
                selected_column = "embedding"
                query_vec = base_vec
        else:
            if column != "embedding":
                print("[info] Query image lacks stored mask features; falling back to base embedding.")
                selected_column = "embedding"

        if query_vec is None:
            query_vec = embedder.embed_image(image_path).astype(np.float32, copy=False)

        if dissimilar:
            hits = db.search_dissimilar(query_vec, top_k=top_k, return_metadata=True, column=selected_column)
        else:
            hits = db.search_similar(query_vec, top_k=top_k, return_metadata=True, column=selected_column)
    return cast(List[SearchHit], hits)


def query_by_text(
    text: str,
    db_cfg: Dict[str, object],
    *,
    top_k: int,
    dissimilar: bool,
    vector_dim: int,
    column: str,
    embedder: EmbedderType,
) -> List[SearchHit]:
    query_vec = embedder.embed_text(text)

    with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
        col = column if column == "embedding" else "embedding"
        if column != "embedding":
            print("[info] Text queries only support image embeddings; ignoring --feature-space.")
        if dissimilar:
            hits = db.search_dissimilar(query_vec, top_k=top_k, return_metadata=True, column=col)
        else:
            hits = db.search_similar(query_vec, top_k=top_k, return_metadata=True, column=col)
    return cast(List[SearchHit], hits)


def query_by_location(
    location_key: str,
    year: int | None,
    db_cfg: Dict[str, object],
    *,
    top_k: int,
    dissimilar: bool,
    vector_dim: int,
    column: str,
) -> List[SearchHit]:
    with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
        rows = db.fetch_embeddings_by_location(location_key)
        if not rows:
            raise ValueError(f"No embeddings stored for location='{location_key}'.")
        if year is None:
            entry: StoredEmbedding = rows[-1]
        else:
            matches = [r for r in rows if r.year == year]
            if not matches:
                available = ", ".join(str(r.year) for r in rows)
                raise ValueError(
                    f"No embedding for location='{location_key}' year={year}. Available years: {available}"
                )
            entry = matches[0]
        selected_column = column
        embedding = entry.embedding.astype(np.float32, copy=False)
        if column == "fusion_embedding" and entry.fusion_embedding is not None:
            embedding = entry.fusion_embedding.astype(np.float32, copy=False)
        elif column == "mask_embedding" and entry.mask_embedding is not None:
            embedding = entry.mask_embedding.astype(np.float32, copy=False)
        elif column == "mask_image_embedding" and entry.mask_image_embedding is not None:
            embedding = entry.mask_image_embedding.astype(np.float32, copy=False)
        elif column != "embedding":
            print("[info] Requested feature space not available for this location; using base embedding.")
            selected_column = "embedding"
        path = entry.image_path
        use_year = entry.year
        print(f"Using stored embedding from {path} (year={use_year}) as the query vector.")
        if dissimilar:
            hits = db.search_dissimilar(embedding, top_k=top_k, return_metadata=True, exclude_path=path, column=selected_column)
        else:
            hits = db.search_similar(embedding, top_k=top_k, return_metadata=True, exclude_path=path, column=selected_column)
    return cast(List[SearchHit], hits)


def analyze_location_changes(
    location_key: str,
    db_cfg: Dict[str, object],
    *,
    year_a: int | None,
    year_b: int | None,
    all_pairs: bool,
    vector_dim: int,
) -> List[ChangeVector]:
    with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
        if year_a is not None and year_b is not None:
            change = db.compare_years(location_key, year_a, year_b)
            if change is None:
                raise ValueError(
                    f"Unable to compute change vector for {location_key} between {year_a} and {year_b}."
                )
            _print_changes(
                f"Change vector for {location_key}: {year_a}->{year_b}",
                [change],
            )
        ranked = db.rank_year_pairs(location_key, consecutive_only=not all_pairs)
        return ranked




def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive queries against the pgvector image retrieval DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usages:

  # External image as query
  python query.py --mode image --query /tmp/example.png --top-k 10

  # Stored location, year constrained, return dissimilar results
  python query.py --mode location --location-key NY_123 --year 2018 --dissimilar

  # Inspect change magnitudes for a stored location
  python query.py --mode delta --location-key NY_123 --all-pairs
        """,
    )
    def _append_suffix(name: str, encoder: str) -> str:
        suffix = f"_{encoder}"
        known = ("_clip", "_eva_clip")
        if name.endswith(known):
            return name
        return name if name.endswith(suffix) else f"{name}{suffix}"

    parser.add_argument("--mode", choices=["image", "text", "location", "delta"], required=True)
    parser.add_argument("--query", type=str, help="Image path or text prompt for image/text modes.")
    parser.add_argument(
        "--location-key",
        dest="location_key",
        type=str,
        help="Stable location identifier (slug without year).",
    )
    parser.add_argument(
        "--image-name",
        dest="location_key",
        type=str,
        help="Deprecated alias for --location-key.",
    )
    parser.add_argument("--year", type=int, help="Year to use for --mode location.")
    parser.add_argument("--year-a", type=int, help="Starting year for --mode delta.")
    parser.add_argument("--year-b", type=int, help="Ending year for --mode delta.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbors to return (image/text/location modes).")
    parser.add_argument("--dissimilar", action="store_true", help="Return the most dissimilar matches instead of similar ones.")
    parser.add_argument("--all-pairs", action="store_true", help="When inspecting changes, consider all year pairs (not only consecutive).")
    parser.add_argument(
        "--feature-space",
        choices=["image", "fusion", "mask", "mask-image"],
        default="image",
        help="Embedding column to query: image (base CLIP), fusion (mask-enhanced), mask (segmentation-only), mask-image (raw mask tile).",
    )
    parser.add_argument("--encoder", choices=["clip", "eva_clip"], default="clip",
                        help="Encoder to use for on-the-fly embeddings.")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32",
                        help="open_clip model name when --encoder=clip.")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k",
                        help="open_clip pretrained tag when --encoder=clip.")
    parser.add_argument("--eva-model", type=str, default="EVA02-CLIP-g-14",
                        help="EVA-CLIP model name (open_clip) when --encoder=eva_clip.")
    parser.add_argument("--eva-pretrained", type=str, default="laion2b_s4b_b79k",
                        help="EVA-CLIP pretrained tag when --encoder=eva_clip.")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                        help="Force device for the encoder (optional).")

    parser.add_argument("--db-name", type=str, default="image_retrieval")
    parser.add_argument("--db-user", type=str, default="postgres")
    parser.add_argument("--db-password", type=str, default="postgres")
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--vector-dim", type=int, default=None,
                        help="Override embedding dimension (auto from encoder when omitted).")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    db_name = _append_suffix(args.db_name, args.encoder)

    db_cfg = {
        "dbname": db_name,
        "user": args.db_user,
        "password": args.db_password,
        "host": args.db_host,
        "port": args.db_port,
    }
    feature_column = _feature_to_column(args.feature_space)
    embedder: Optional[EmbedderType] = None
    if args.mode in {"image", "text"} or args.vector_dim is None:
        embedder = _make_embedder(
            args.encoder,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            eva_model=args.eva_model,
            eva_pretrained=args.eva_pretrained,
            device=args.device,
        )
    resolved_vector_dim = args.vector_dim or (
        embedder.embedding_dim if embedder is not None else 512
    )

    if args.mode in {"image", "text"} and not args.query:
        parser.error("--query is required for image/text modes")
    if args.mode in {"location", "delta"} and not args.location_key:
        parser.error("--location-key is required for location/delta modes")

    if args.mode == "image":
        if embedder is None:
            raise RuntimeError("Embedder not initialised for image mode.")
        hits = query_by_image(
            args.query,
            db_cfg,
            top_k=args.top_k,
            dissimilar=args.dissimilar,
            vector_dim=resolved_vector_dim,
            column=feature_column,
            embedder=embedder,
        )
        title = "Most dissimilar" if args.dissimilar else "Most similar"
        _print_hits(f"{title} results for query image {args.query}", hits)

    elif args.mode == "text":
        if embedder is None:
            raise RuntimeError("Embedder not initialised for text mode.")
        hits = query_by_text(
            args.query,
            db_cfg,
            top_k=args.top_k,
            dissimilar=args.dissimilar,
            vector_dim=resolved_vector_dim,
            column=feature_column,
            embedder=embedder,
        )
        title = "Most dissimilar" if args.dissimilar else "Most similar"
        _print_hits(f"{title} results for text query '{args.query}'", hits)

    elif args.mode == "location":
        hits = query_by_location(
            args.location_key,
            args.year,
            db_cfg,
            top_k=args.top_k,
            dissimilar=args.dissimilar,
            vector_dim=resolved_vector_dim,
            column=feature_column,
        )
        title = "Most dissimilar" if args.dissimilar else "Most similar"
        year_desc = f"year={args.year}" if args.year is not None else "latest year"
        _print_hits(f"{title} results for stored location {args.location_key} ({year_desc})", hits)

    elif args.mode == "delta":
        ranked = analyze_location_changes(
            args.location_key,
            db_cfg,
            year_a=args.year_a,
            year_b=args.year_b,
            all_pairs=args.all_pairs,
            vector_dim=resolved_vector_dim,
        )
        if ranked:
            heading = "Change ranking (largest cosine distance first)"
            _print_changes(heading, ranked, limit=20)
    else:  # pragma: no cover - argparse prevents
        parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
