"""
Generate CLIP embeddings for images and store in DuckDB.

This script generates CLIP embeddings for street imagery and stores them
in the streettransformer database. It can also compute change vectors between years.

Usage:
    # Generate embeddings from a directory of images
    python scripts/generate_embeddings.py --universe lion --year 2020 --image-dir /path/to/images

    # Generate for multiple years
    python scripts/generate_embeddings.py --universe lion --years 2018 2020 2022 --image-dir /path/to/images

    # Compute change vectors between consecutive years
    python scripts/generate_embeddings.py --universe lion --compute-changes

    # Use custom CLIP model
    python scripts/generate_embeddings.py --universe lion --year 2020 --model ViT-L-14 --pretrained laion2b_s34b_b79k

Requirements:
    pip install open-clip-torch torch pillow numpy pandas tqdm

Example workflow:
    1. Organize images in directories by year (optional)
    2. Generate embeddings with this script
    3. Query similar images using StreetTransformer query classes
"""

from argparse import ArgumentParser
from pathlib import Path
import logging
from typing import Any
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from streettransformer import STConfig, EmbeddingDB, ImageEmbedding

logger = logging.getLogger('generate_embeddings')


def load_clip_model(model_name: str = 'ViT-B-32', pretrained: str = 'openai'):
    """Load CLIP model for embedding generation.

    Args:
        model_name: CLIP model architecture
        pretrained: Pre-trained weights (e.g., 'openai', 'laion2b_s34b_b79k')

    Returns:
        Tuple of (model, preprocess, device)
    """
    try:
        import open_clip
        import torch
    except ImportError:
        raise ImportError(
            "open_clip_torch and torch are required. "
            "Install with: pip install open-clip-torch torch"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {device}")

    # Use force_quick_gelu for OpenAI weights
    force_quick_gelu = (pretrained == "openai")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        force_quick_gelu=force_quick_gelu
    )
    model = model.to(device)
    model.eval()

    logger.info(f"CLIP model loaded successfully")
    return model, preprocess, device


def get_images_from_directory(
    image_dir: Path,
    year: int | None = None,
    extensions: tuple = ('.jpg', '.jpeg', '.png')
) -> pd.DataFrame:
    """Get list of images from a directory.

    Args:
        image_dir: Directory containing images
        year: Optional year filter (uses directory name if None)
        extensions: Valid image extensions

    Returns:
        DataFrame with columns: location_id, year, file_path_abs
    """
    images = []
    image_paths = []

    # Recursively find all image files
    for ext in extensions:
        image_paths.extend(image_dir.rglob(f'*{ext}'))

    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    for img_path in image_paths:
        # Extract location_id from filename (assumes format: locationid_*.jpg)
        # Adjust this logic based on your naming convention
        filename = img_path.stem

        # Try to extract location_id from filename
        try:
            # Common format: "123_2020.jpg" or "location_123_other.jpg"
            parts = filename.split('_')
            location_id = None

            for part in parts:
                if part.isdigit():
                    location_id = int(part)
                    break

            if location_id is None:
                logger.warning(f"Could not extract location_id from {filename}, using hash")
                location_id = hash(filename) % (10 ** 8)

        except Exception as e:
            logger.warning(f"Error parsing {filename}: {e}, using hash")
            location_id = hash(filename) % (10 ** 8)

        # Determine year (from argument, directory name, or filename)
        img_year = year
        if img_year is None:
            # Try to extract year from parent directory or filename
            if img_path.parent.name.isdigit():
                img_year = int(img_path.parent.name)
            else:
                # Try to find year in filename
                for part in filename.split('_'):
                    if part.isdigit() and len(part) == 4 and 1900 < int(part) < 2100:
                        img_year = int(part)
                        break

                if img_year is None:
                    logger.warning(f"Could not determine year for {img_path}, skipping")
                    continue

        images.append({
            'location_id': location_id,
            'year': img_year,
            'file_path_abs': str(img_path.absolute())
        })

    df = pd.DataFrame(images)
    logger.info(f"Processed {len(df)} valid images")
    return df


def generate_embeddings_for_year(
    config: STConfig,
    year: int,
    image_dir: Path,
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 64,
    skip_existing: bool = True
) -> int:
    """Generate embeddings for all images in a specific year.

    Args:
        config: StreetTransformer configuration
        year: Year to process
        image_dir: Directory containing images
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device for inference
        batch_size: Batch size for processing
        skip_existing: Skip images that already have embeddings

    Returns:
        Number of embeddings generated
    """
    import torch
    from PIL import Image

    # Get images for this year
    images_df = get_images_from_directory(image_dir, year=year)

    if images_df.empty:
        logger.warning(f"No images found for year {year}")
        return 0

    # Create location keys (format: location_{location_id})
    images_df['location_key'] = images_df['location_id'].apply(lambda x: f"location_{x}")

    # Setup embedding database
    db = EmbeddingDB(config)

    # Check which images already have embeddings
    if skip_existing:
        try:
            existing = db.fetch_embeddings_by_year(year)
            existing_keys = set(existing['location_key'].values)
            images_df = images_df[~images_df['location_key'].isin(existing_keys)]
            logger.info(f"Found {len(existing_keys)} existing embeddings, processing {len(images_df)} new images")
        except Exception as e:
            logger.warning(f"Could not check existing embeddings: {e}")

    if images_df.empty:
        logger.info(f"No new images to process for year {year}")
        return 0

    # Process in batches
    embeddings_to_insert = []
    total_processed = 0

    with tqdm(total=len(images_df), desc=f"Generating embeddings for {year}") as pbar:
        for i in range(0, len(images_df), batch_size):
            batch_df = images_df.iloc[i:i+batch_size]

            # Load and preprocess images
            images = []
            valid_rows = []

            for row in batch_df.itertuples():
                image_path = Path(row.file_path_abs)

                try:
                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}")
                        continue

                    img = Image.open(image_path).convert('RGB')
                    images.append(preprocess(img))
                    valid_rows.append(row)

                except Exception as e:
                    logger.error(f"Error loading {image_path}: {e}")
                    continue

            if not images:
                pbar.update(len(batch_df))
                continue

            # Generate embeddings
            try:
                images_tensor = torch.stack(images).to(device)

                with torch.no_grad():
                    embeddings = model.encode_image(images_tensor)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    embeddings_np = embeddings.cpu().numpy()

                # Create ImageEmbedding objects
                for row, embedding in zip(valid_rows, embeddings_np):
                    emb_obj = ImageEmbedding(
                        location_id=int(row.location_id),
                        location_key=row.location_key,
                        year=int(row.year),
                        image_path=str(row.file_path_abs),
                        embedding=embedding
                    )
                    embeddings_to_insert.append(emb_obj)

                total_processed += len(valid_rows)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")

            pbar.update(len(batch_df))

            # Insert in batches to avoid memory issues
            if len(embeddings_to_insert) >= 1000:
                db.insert_embeddings(embeddings_to_insert, on_conflict='replace')
                logger.info(f"Inserted {len(embeddings_to_insert)} embeddings")
                embeddings_to_insert = []

    # Insert remaining embeddings
    if embeddings_to_insert:
        db.insert_embeddings(embeddings_to_insert, on_conflict='replace')
        logger.info(f"Inserted {len(embeddings_to_insert)} embeddings")

    logger.info(f"Generated {total_processed} embeddings for year {year}")
    return total_processed


def compute_all_change_vectors(
    config: STConfig,
    years: list[int] | None = None,
    consecutive_only: bool = False
) -> None:
    """Compute change vectors between years.

    Args:
        config: StreetTransformer configuration
        years: List of years (auto-detect if None)
        consecutive_only: Only compute changes for consecutive years
    """
    db = EmbeddingDB(config)

    # Auto-detect years if not provided
    if years is None:
        years = db.get_years()
        logger.info(f"Auto-detected years: {years}")

    if len(years) < 2:
        logger.warning("Need at least 2 years to compute change vectors")
        return

    years = sorted(years)

    # Compute change vectors
    if consecutive_only:
        pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
    else:
        pairs = [(y1, y2) for i, y1 in enumerate(years) for y2 in years[i+1:]]

    logger.info(f"Computing {len(pairs)} change vector pairs")

    for year_from, year_to in pairs:
        logger.info(f"Computing change vectors: {year_from} -> {year_to}")
        db.compute_change_vectors(
            year_from=year_from,
            year_to=year_to,
            normalize=True
        )


def main():
    parser = ArgumentParser(description="Generate CLIP embeddings for street imagery")
    parser.add_argument('--universe', '-u', type=str, required=True, help='Universe name')
    parser.add_argument('--db', type=str, help='Database path (overrides ST_DATABASE_PATH env var)')
    parser.add_argument('--image-dir', type=Path, help='Directory containing images')
    parser.add_argument('--years', '-y', nargs='+', type=int, help='Multiple years to process')
    parser.add_argument('--model', type=str, default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pre-trained weights')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--compute-changes', action='store_true', help='Compute change vectors')
    parser.add_argument('--all-pairs', action='store_true', help='Compute all year pairs (not just consecutive)')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip existing embeddings')
    parser.add_argument('--log-file', type=Path, help='Path to log file')

    args = parser.parse_args()

    # Setup logging
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # Add file handler if log file specified
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(log_format)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")

    # Create config
    config = STConfig(
        database_path=args.db,
        universe_name=args.universe
    )

    # Initialize database schema
    db = EmbeddingDB(config)
    db.setup_schema(create_index=True)
    logger.info(f"Database initialized: {config.database_path}")

    # Determine years to process
    years_to_process = args.years or []

    # Load CLIP model if generating embeddings
    if years_to_process and args.image_dir:
        if not args.image_dir.exists():
            logger.error(f"Image directory does not exist: {args.image_dir}")
            return

        model, preprocess, device = load_clip_model(args.model, args.pretrained)

        # Generate embeddings for each year
        for year in years_to_process:
            logger.info(f"Processing year {year}")
            count = generate_embeddings_for_year(
                config=config,
                year=year,
                image_dir=args.image_dir,
                model=model,
                preprocess=preprocess,
                device=device,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing
            )
            logger.info(f"Completed year {year}: {count} embeddings generated")

    # Compute change vectors if requested
    if args.compute_changes:
        compute_all_change_vectors(
            config=config,
            years=years_to_process if years_to_process else None,
            consecutive_only=not args.all_pairs
        )

    logger.info("Embedding generation complete!")


if __name__ == '__main__':
    main()
