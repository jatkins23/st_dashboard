"""
Generate CLIP embeddings for multiple media types and store in DuckDB.

This script generates CLIP embeddings for street imagery (images, masks, side-by-side)
and stores them in the streettransformer database. It can also compute change vectors between years.

Usage:
    # Load file paths from database (auto-detects years and media types)
    python scripts/generate_embeddings.py --universe nyc --from-db

    # Load from database for specific years
    python scripts/generate_embeddings.py --universe nyc --from-db --years 2020 2022

    # Load from database for specific media types
    python scripts/generate_embeddings.py --universe nyc --from-db --media-types image mask

    # Generate embeddings from filesystem for images only
    python scripts/generate_embeddings.py --universe lion --years 2020 --image-dir /path/to/images

    # Generate embeddings from filesystem for multiple media types
    python scripts/generate_embeddings.py --universe lion --years 2020 \
        --image-dir /path/to/images \
        --mask-dir /path/to/masks \
        --sidebyside-dir /path/to/sidebyside

    # Process specific media types from multiple directories
    python scripts/generate_embeddings.py --universe lion --years 2020 \
        --image-dir /path/to/images \
        --mask-dir /path/to/masks \
        --media-types image mask

    # Generate for multiple years
    python scripts/generate_embeddings.py --universe lion --years 2018 2020 2022 --image-dir /path/to/images

    # Compute change vectors for all year pairs (default)
    python scripts/generate_embeddings.py --universe lion --compute-changes

    # Compute change vectors for consecutive years only
    python scripts/generate_embeddings.py --universe lion --compute-changes --consecutive-only

    # Use custom CLIP model
    python scripts/generate_embeddings.py --universe lion --years 2020 --model ViT-L-14 --pretrained laion2b_s34b_b79k

    # Use 8 parallel workers for faster image loading
    python scripts/generate_embeddings.py --universe nyc --from-db --num-workers 8

Requirements:
    pip install open-clip-torch torch pillow numpy pandas tqdm

Performance Tuning:
    - --batch-size: GPU processing batch size (default: 64, increase for more GPU memory)
    - --num-workers: Parallel image loading workers (default: 4, try 4-8 for best speed)
    - More workers = faster image loading but more CPU/RAM usage

Example workflow (database mode):
    1. Ensure location_year_files table is populated with file paths
    2. Run script with --from-db to auto-detect years and media types
    3. Query similar images using StreetTransformer query classes

Example workflow (filesystem mode):
    1. Organize images/masks/sidebyside in year subdirectories (e.g., images/2020/, images/2022/)
    2. Generate embeddings for all media types with this script
    3. Query similar images using StreetTransformer query classes with media_type filters
"""

from argparse import ArgumentParser
from pathlib import Path
import logging
from typing import Any
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from streettransformer import STConfig, EmbeddingDB, MediaEmbedding

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


class ImageDataset:
    """Dataset for loading images with CLIP preprocessing.

    Enables parallel image loading using DataLoader workers.
    """
    def __init__(self, df: pd.DataFrame, preprocess: Any):
        """Initialize dataset.

        Args:
            df: DataFrame with columns: location_id, location_key, year, media_type, file_path_abs
            preprocess: CLIP preprocessing transform
        """
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and preprocess image at index.

        Returns:
            Tuple of (image_tensor, metadata_dict) or None if error
        """
        from PIL import Image

        row = self.df.iloc[idx]

        try:
            img = Image.open(row['file_path_abs']).convert('RGB')
            img_tensor = self.preprocess(img)

            metadata = {
                'location_id': row['location_id'],
                'location_key': row['location_key'],
                'year': int(row['year']),
                'media_type': row['media_type'],
                'file_path_abs': str(row['file_path_abs'])
            }

            return img_tensor, metadata

        except Exception as e:
            # Return None for failed loads (will be filtered out)
            logger.warning(f"Error loading {row['file_path_abs']}: {e}")
            return None


def collate_fn(batch):
    """Custom collate function to filter out failed image loads.

    Args:
        batch: List of (image_tensor, metadata) tuples or None values

    Returns:
        Tuple of (stacked_tensors, list_of_metadata)
    """
    import torch

    # Filter out None values (failed loads)
    batch = [item for item in batch if item is not None]

    if not batch:
        return None, []

    # Separate images and metadata
    images = [item[0] for item in batch]
    metadata = [item[1] for item in batch]

    # Stack images into a batch tensor
    images_tensor = torch.stack(images)

    return images_tensor, metadata




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

    # Determine search directory
    if year is not None:
        # When year is specified, only search in that year's subdirectory
        search_dir = image_dir / str(year)
        if not search_dir.exists():
            logger.warning(f"Year subdirectory does not exist: {search_dir}")
            return pd.DataFrame(columns=['location_id', 'year', 'file_path_abs'])
    else:
        # When year is not specified, search entire directory
        search_dir = image_dir

    # Recursively find all image files
    for ext in extensions:
        image_paths.extend(search_dir.rglob(f'*{ext}'))

    logger.info(f"Found {len(image_paths)} images in {search_dir}")

    for img_path in image_paths:
        # Extract location_id from filename (assumes format: locationid_*.jpg)
        # Adjust this logic based on your naming convention
        location_id = img_path.stem    

        # Determine year (from argument, directory name, or filename)
        img_year = year
        if img_year is None:
            # Try to extract year from parent directory or filename
            if img_path.parent.name.isdigit():
                img_year = int(img_path.parent.name)
            else:
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




def get_media_directories(args: Any) -> dict[str, Path]:
    """Get media directories from command line arguments and validate they exist.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary mapping media type to directory path

    Raises:
        ValueError: If a specified directory doesn't exist
    """
    media_dirs = {}

    if args.image_dir:
        if not args.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {args.image_dir}")
        media_dirs['image'] = args.image_dir

    if args.mask_dir:
        if not args.mask_dir.exists():
            raise ValueError(f"Mask directory does not exist: {args.mask_dir}")
        media_dirs['mask'] = args.mask_dir

    if args.sidebyside_dir:
        if not args.sidebyside_dir.exists():
            raise ValueError(f"Side-by-side directory does not exist: {args.sidebyside_dir}")
        media_dirs['sidebyside'] = args.sidebyside_dir

    return media_dirs

def gather_all_media_files(config: STConfig, args: Any) -> dict[str, dict[int, pd.DataFrame]]:
    """Gather all media files to process from database or filesystem.

    Args:
        config: StreetTransformer configuration
        args: Parsed command line arguments

    Returns:
        Dictionary mapping media_type -> year -> DataFrame with columns:
            location_id, location_key, year, media_type, file_path_abs

    Raises:
        ValueError: If configuration is invalid
    """
    from streettransformer.db.database import get_connection

    files_by_media_type = {}

    if args.from_db:
        # Gather from database
        with get_connection(config.database_path, read_only=True) as con:
            # Determine which media types to query
            if args.media_types:
                media_types = args.media_types
            else:
                media_types_df = con.execute(f"""
                    SELECT DISTINCT file_type
                    FROM {config.universe_name}.location_year_files
                    WHERE exists = true
                    ORDER BY file_type
                """).df()
                media_types = media_types_df['file_type'].tolist()
                logger.info(f"Auto-detected media types from database: {media_types}")

            # Build year filter
            year_filter = ""
            if args.years:
                year_list = ','.join(map(str, args.years))
                year_filter = f"AND year IN ({year_list})"

            # Query all files for each media type
            for media_type in media_types:
                query = f"""
                    SELECT
                        location_id,
                        'location_' || CAST(location_id AS VARCHAR) as location_key,
                        year,
                        '{media_type}' as media_type,
                        file_path_abs
                    FROM {config.universe_name}.location_year_files
                    WHERE file_type = '{media_type}'
                    AND exists = true
                    {year_filter}
                    ORDER BY year, location_id
                """
                df = con.execute(query).df()
                if not df.empty:
                    # Split by year
                    files_by_media_type[media_type] = {}
                    for year in df['year'].unique():
                        year_df = df[df['year'] == year].copy()
                        files_by_media_type[media_type][int(year)] = year_df

                    logger.info(f"Found {len(df)} {media_type} files across {len(files_by_media_type[media_type])} years")

    else:
        # Gather from filesystem
        if not args.years:
            raise ValueError("--years required when using filesystem mode")

        media_dirs = get_media_directories(args)

        # Determine which media types to process
        if args.media_types:
            media_types = args.media_types
            for media_type in media_types:
                if media_type not in media_dirs:
                    raise ValueError(f"Media type '{media_type}' specified but no directory provided (use --{media_type}-dir)")
        else:
            media_types = list(media_dirs.keys())

        # Gather files from each directory
        for media_type in media_types:
            media_dir = media_dirs[media_type]
            files_by_year = {}

            for year in args.years:
                year_df = get_images_from_directory(media_dir, year=year)
                if not year_df.empty:
                    # Add media_type and location_key columns
                    year_df['media_type'] = media_type
                    year_df['location_key'] = year_df['location_id'].apply(lambda x: f"location_{x}")
                    files_by_year[year] = year_df

            if files_by_year:
                files_by_media_type[media_type] = files_by_year
                total = sum(len(df) for df in files_by_year.values())
                logger.info(f"Found {total} {media_type} files across {len(files_by_year)} years")

    return files_by_media_type


def filter_existing_embeddings(
    config: STConfig,
    files_by_media_type: dict[str, dict[int, pd.DataFrame]]
) -> dict[str, dict[int, pd.DataFrame]]:
    """Filter out files that already have embeddings.

    Args:
        config: StreetTransformer configuration
        files_by_media_type: Dictionary mapping media_type -> year -> file DataFrames

    Returns:
        Filtered dictionary with only files that need embeddings
    """
    db = EmbeddingDB(config)
    filtered = {}

    for media_type, years_dict in files_by_media_type.items():
        filtered_years = {}

        for year, files_df in years_dict.items():
            if files_df.empty:
                continue

            # Get existing embeddings for this media type and year
            try:
                existing = db.fetch_embeddings_by_year(year, media_type=media_type)

                if not existing.empty:
                    existing_keys = set(existing['location_key'].values)

                    # Filter out existing
                    new_files = files_df[~files_df['location_key'].isin(existing_keys)]

                    if not new_files.empty:
                        filtered_years[year] = new_files
                        logger.info(
                            f"{media_type} {year}: {len(existing_keys)} already embedded, "
                            f"{len(new_files)} new files to process"
                        )
                    else:
                        logger.info(f"{media_type} {year}: All files already embedded")
                else:
                    # No existing embeddings, process all
                    filtered_years[year] = files_df
                    logger.info(f"{media_type} {year}: {len(files_df)} new files to process")

            except Exception as e:
                logger.warning(f"Could not check existing embeddings for {media_type} {year}: {e}")
                filtered_years[year] = files_df

        if filtered_years:
            filtered[media_type] = filtered_years

    return filtered


def process_all_media(
    config: STConfig,
    files_by_media_type: dict[str, dict[int, pd.DataFrame]],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> dict[str, int]:
    """Process all media files and generate embeddings using parallel DataLoader.

    Args:
        config: StreetTransformer configuration
        files_by_media_type: Dictionary mapping media_type -> year -> file DataFrames
        model: CLIP model
        preprocess: Image preprocessing function
        device: Device for inference
        batch_size: Batch size for processing
        num_workers: Number of parallel workers for image loading

    Returns:
        Dictionary mapping media_type to count of embeddings generated
    """
    import torch
    from torch.utils.data import DataLoader

    db = EmbeddingDB(config)
    results = {}

    for media_type, years_dict in files_by_media_type.items():
        total_media_type_count = sum(len(df) for df in years_dict.values())
        logger.info(f"Processing {total_media_type_count} {media_type} files across {len(years_dict)} years")

        embeddings_to_insert = []
        total_processed = 0

        with tqdm(total=total_media_type_count, desc=f"{media_type.capitalize()}", unit="img") as pbar:
            for year in sorted(years_dict.keys()):
                files_df = years_dict[year]

                if files_df.empty:
                    continue

                # Create dataset and dataloader for parallel image loading
                dataset = ImageDataset(files_df, preprocess)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    pin_memory=(device == "cuda"),  # Faster GPU transfer
                    prefetch_factor=2 if num_workers > 0 else None  # Prefetch 2 batches per worker
                )

                for batch_images, batch_metadata in dataloader:
                    if batch_images is None or len(batch_metadata) == 0:
                        # Empty batch (all images failed to load)
                        continue

                    try:
                        # Transfer to device
                        batch_images = batch_images.to(device)

                        # Generate embeddings
                        with torch.no_grad():
                            embeddings = model.encode_image(batch_images)
                            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                            embeddings_np = embeddings.cpu().numpy()

                        # Create MediaEmbedding objects
                        for metadata, embedding in zip(batch_metadata, embeddings_np):
                            emb_obj = MediaEmbedding(
                                location_id=metadata['location_id'],
                                location_key=metadata['location_key'],
                                year=metadata['year'],
                                media_type=metadata['media_type'],
                                path=metadata['file_path_abs'],
                                embedding=embedding,
                                stats=None
                            )
                            embeddings_to_insert.append(emb_obj)

                        total_processed += len(batch_metadata)

                    except Exception as e:
                        logger.error(f"Error generating embeddings for batch: {e}")

                    # Insert in batches to avoid memory issues
                    if len(embeddings_to_insert) >= 1000:
                        db.insert_embeddings(embeddings_to_insert, on_conflict='replace')
                        embeddings_to_insert = []

                    pbar.update(len(batch_metadata))

        # Insert remaining embeddings
        if embeddings_to_insert:
            db.insert_embeddings(embeddings_to_insert, on_conflict='replace')

        results[media_type] = total_processed
        logger.info(f"Completed {media_type}: {total_processed} embeddings generated")

    return results


def compute_all_change_vectors(
    config: STConfig,
    years: list[int] | None = None,
    media_types: list[str] | None = None,
    consecutive_only: bool = False
) -> None:
    """Compute change vectors between years.

    Args:
        config: StreetTransformer configuration
        years: List of years (auto-detect if None)
        media_types: List of media types to compute changes for (auto-detect if None)
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

    # Auto-detect media types if not provided
    if media_types is None:
        media_types = db.get_media_types()
        logger.info(f"Auto-detected media types: {media_types}")

    if not media_types:
        logger.warning("No media types found")
        return

    years = sorted(years)

    # Compute change vectors
    if consecutive_only:
        pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
    else:
        pairs = [(y1, y2) for i, y1 in enumerate(years) for y2 in years[i+1:]]

    logger.info(f"Computing {len(pairs)} change vector pairs for {len(media_types)} media types")

    for media_type in media_types:
        for year_from, year_to in pairs:
            logger.info(f"Computing change vectors: {year_from} -> {year_to} ({media_type})")
            db.compute_change_vectors(
                year_from=year_from,
                year_to=year_to,
                media_type=media_type,
                normalize=True
            )


def main():
    parser = ArgumentParser(description="Generate CLIP embeddings for street imagery")
    parser.add_argument('--universe', '-u', type=str, required=True, help='Universe name')
    parser.add_argument('--db', type=str, help='Database path (overrides ST_DATABASE_PATH env var)')
    parser.add_argument('--from-db', action='store_true', help='Load file paths from location_year_files table instead of filesystem')
    parser.add_argument('--image-dir', type=Path, help='Directory containing images')
    parser.add_argument('--mask-dir', type=Path, help='Directory containing masks')
    parser.add_argument('--sidebyside-dir', type=Path, help='Directory containing side-by-side images')
    parser.add_argument('--years', '-y', nargs='+', type=int, help='Multiple years to process')
    parser.add_argument('--media-types', nargs='+', type=str, choices=['image', 'mask', 'sidebyside'], help='Media types to process (default: auto-detect from provided directories or database)')
    parser.add_argument('--model', type=str, default='ViT-B-32', help='CLIP model name')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pre-trained weights')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=4, help='Number of parallel workers for image loading (default: 4)')
    parser.add_argument('--compute-changes', action='store_true', help='Compute change vectors')
    parser.add_argument('--consecutive-only', action='store_true', help='Only compute consecutive year pairs (default: compute all pairs)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing embeddings')
    parser.add_argument('--dry-run', action='store_true', help='Preview what would be processed without actually processing')
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

    # Phase 1: Gather all media files
    try:
        files_by_media_type = gather_all_media_files(config, args)
    except ValueError as e:
        logger.error(str(e))
        return

    if not files_by_media_type:
        logger.warning("No media files found to process")
        return

    # Log summary
    total_files = sum(
        len(df)
        for years_dict in files_by_media_type.values()
        for df in years_dict.values()
    )
    logger.info(f"Found {total_files} total files across {len(files_by_media_type)} media types")

    # Log detailed breakdown by media type and year
    for media_type, years_dict in files_by_media_type.items():
        years_summary = {year: len(df) for year, df in years_dict.items()}
        logger.info(f"  {media_type}: {years_summary}")

    # Phase 2: Filter existing embeddings if requested
    if args.skip_existing:
        files_by_media_type = filter_existing_embeddings(config, files_by_media_type)

        if not files_by_media_type:
            logger.info("All files already have embeddings")
            # Still compute change vectors if requested
            if args.compute_changes and not args.dry_run:
                compute_all_change_vectors(
                    config=config,
                    consecutive_only=args.consecutive_only
                )
            return

    # Dry-run mode: print summary and test one embedding per type/year
    if args.dry_run:
        import torch
        from PIL import Image

        logger.info("DRY RUN MODE - Testing pipeline without writing to database")
        logger.info("=" * 60)

        total_to_process = sum(
            len(df)
            for years_dict in files_by_media_type.values()
            for df in years_dict.values()
        )

        logger.info(f"Would process {total_to_process} total files")

        for media_type, years_dict in files_by_media_type.items():
            total_media_count = sum(len(df) for df in years_dict.values())
            logger.info(f"\n{media_type.upper()}: {total_media_count} files across {len(years_dict)} years")

            for year in sorted(years_dict.keys()):
                df = years_dict[year]
                logger.info(f"  {year}: {len(df)} files")

                # Show first few file paths as examples
                sample_paths = df['file_path_abs'].head(3).tolist()
                for path in sample_paths:
                    logger.info(f"    - {path}")
                if len(df) > 3:
                    logger.info(f"    ... and {len(df) - 3} more")

        logger.info("=" * 60)

        # Show what change vectors would be computed
        if args.compute_changes:
            all_years = set()
            for years_dict in files_by_media_type.values():
                all_years.update(years_dict.keys())

            if len(all_years) >= 2:
                years = sorted(all_years)
                if args.consecutive_only:
                    pairs = [(years[i], years[i+1]) for i in range(len(years)-1)]
                else:
                    pairs = [(y1, y2) for i, y1 in enumerate(years) for y2 in years[i+1:]]

                logger.info(f"\nWould compute change vectors for {len(pairs)} year pairs:")
                for year_from, year_to in pairs:
                    logger.info(f"  {year_from} -> {year_to}")

        # Load CLIP model and test one embedding per type/year
        logger.info("\n" + "=" * 60)
        logger.info("Testing CLIP embedding pipeline (one sample per type/year)")
        logger.info("=" * 60)

        model, preprocess, device = load_clip_model(args.model, args.pretrained)

        for media_type, years_dict in files_by_media_type.items():
            logger.info(f"\n{media_type.upper()} samples:")

            for year in sorted(years_dict.keys()):
                df = years_dict[year]

                if df.empty:
                    continue

                # Get first file from this type/year combination
                sample_row = df.iloc[0]
                sample_path = sample_row['file_path_abs']

                try:
                    # Load and preprocess image
                    img = Image.open(sample_path).convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(device)

                    # Generate embedding
                    with torch.no_grad():
                        embedding = model.encode_image(img_tensor)
                        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embedding_np = embedding.cpu().numpy()[0]

                    # Print embedding info
                    logger.info(f"  {year}:")
                    logger.info(f"    File: {sample_path}")
                    logger.info(f"    Location: {sample_row['location_key']}")
                    logger.info(f"    Embedding shape: {embedding_np.shape}")
                    logger.info(f"    Embedding norm: {np.linalg.norm(embedding_np):.6f}")
                    logger.info(f"    First 10 values: {embedding_np[:10]}")
                    logger.info(f"    Stats: min={embedding_np.min():.6f}, max={embedding_np.max():.6f}, mean={embedding_np.mean():.6f}")

                except Exception as e:
                    logger.error(f"  {year}: Failed to process {sample_path}: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("Dry run complete. Use without --dry-run to actually process and save all files.")
        return

    # Phase 3: Load CLIP model and process all files
    model, preprocess, device = load_clip_model(args.model, args.pretrained)

    results = process_all_media(
        config=config,
        files_by_media_type=files_by_media_type,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Log results
    total_generated = sum(results.values())
    logger.info(f"Generated {total_generated} total embeddings")
    for media_type, count in results.items():
        logger.info(f"  {media_type}: {count} embeddings")

    # Compute change vectors if requested
    if args.compute_changes:
        # Extract years and media types from what we just processed
        all_years = set()
        for years_dict in files_by_media_type.values():
            all_years.update(years_dict.keys())

        compute_all_change_vectors(
            config=config,
            years=sorted(all_years) if all_years else None,
            media_types=list(files_by_media_type.keys()) if files_by_media_type else None,
            consecutive_only=args.consecutive_only
        )

    logger.info("Embedding generation complete!")


if __name__ == '__main__':
    main()
