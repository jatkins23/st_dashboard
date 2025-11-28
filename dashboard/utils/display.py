"""Result enrichment utilities for adding street names and image paths."""

import logging
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: Path, max_width: int = 400) -> str:
    """Convert image to base64 string for embedding in HTML.

    Args:
        image_path: Path to image file
        max_width: Maximum width for resizing

    Returns:
        Base64 encoded image string
    """
    try:
        if not image_path.exists():
            return None

        img = Image.open(image_path)

        # Resize if too large
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def enrich_results_with_streets(results: pd.DataFrame, db_connection, universe_name: str) -> pd.DataFrame:
    """Enrich results with street names from locations table.

    Args:
        results: DataFrame with search results
        db_connection: Active database connection context manager
        universe_name: Name of universe

    Returns:
        DataFrame with additional_streets column
    """
    if results.empty:
        print('results empty')
        return results

    try:
        # Build column list dynamically based on what's in results
        base_cols = ['r.location_id', 'r.location_key', 'r.similarity', 'r.image_path']

        # Check for year columns
        if 'year' in results.columns:
            base_cols.append('r.year')
        if 'year_from' in results.columns:
            base_cols.extend(['r.year_from', 'r.year_to'])

        col_list = ',\n                    '.join(base_cols)

        with db_connection as con:
            # Register results as temp table
            con.register('_temp_results', results)

            # Join with locations table - explicitly select columns to avoid array issues
            enriched = con.execute(f"""
                SELECT
                    {col_list},
                    l.street1,
                    l.street2,
                    l.additional_streets
                FROM _temp_results r
                LEFT JOIN {universe_name}.locations l
                    ON r.location_id = l.location_id
            """).df()

            print(enriched)

            return enriched
    except Exception as e:
        logger.error(f"Error enriching results with streets: {e}")
        return results


def enrich_change_results_with_images(results: pd.DataFrame, db_connection, universe_name: str) -> pd.DataFrame:
    """Enrich change results with image paths for both years.

    Args:
        results: DataFrame with change search results (has year_from, year_to)
        db_connection: Active database connection context manager
        universe_name: Name of universe

    Returns:
        DataFrame with image_path_from and image_path_to columns
    """
    if results.empty:
        return results

    try:
        with db_connection as con:
            # Register results as temp table
            con.register('_temp_change_results', results)

            # Join with image_embeddings to get both image paths
            enriched = con.execute(f"""
                SELECT
                    r.location_id,
                    r.location_key,
                    r.year_from,
                    r.year_to,
                    r.similarity,
                    e_from.image_path as image_path_from,
                    e_to.image_path as image_path_to,
                    l.additional_streets
                FROM _temp_change_results r
                LEFT JOIN {universe_name}.image_embeddings e_from
                    ON r.location_id = e_from.location_id
                    AND r.year_from = e_from.year
                LEFT JOIN {universe_name}.image_embeddings e_to
                    ON r.location_id = e_to.location_id
                    AND r.year_to = e_to.year
                LEFT JOIN {universe_name}.locations l
                    ON r.location_id = l.location_id
            """).df()

            return enriched
    except Exception as e:
        logger.error(f"Error enriching change results with images: {e}")
        return results
