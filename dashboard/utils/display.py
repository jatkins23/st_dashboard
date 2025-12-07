"""Result enrichment utilities for adding street names and image paths."""

import logging
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

NO_STREETS: bool = True
ADDTNL_STREETS: bool = True
BOROUGH: bool = True

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


def encode_pdf_to_base64(pdf_path: Path, page_num: int = 0, max_width: int = 400) -> str:
    """Convert PDF page to base64 string for embedding in HTML.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number to convert (0-indexed)
        max_width: Maximum width for resizing

    Returns:
        Base64 encoded image string
    """
    try:
        if not pdf_path.exists():
            return None

        import fitz  # PyMuPDF

        # Open PDF and get the specified page
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            logger.warning(f"Page {page_num} not found in {pdf_path}")
            return None

        page = doc[page_num]

        # Render page to image (matrix for resolution)
        zoom = 2.0  # Higher zoom for better quality
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

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
    except ImportError:
        logger.error(f"PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
        return None
    except Exception as e:
        logger.error(f"Error encoding PDF {pdf_path}: {e}")
        return None

def _querify_list(col_list: list[str], prefix: str, indent: int = 0, newline: bool = True):
    prefixed_list = [f'{prefix}.{c}' for c in col_list]
    sep = ''
    if newline:
        sep = sep + '\n'
    sep = sep + ' ' * indent

    return sep.join(prefixed_list)

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
        results_cols = results.columns.to_list()
        results_cols_query = _querify_list(results_cols, prefix = 'r', indent=20)

        streets_cols = []
        if not NO_STREETS:
            streets_cols = streets_cols.extend(['street1','street2'])
        if ADDTNL_STREETS:
            streets_cols = streets_cols.append('additional_streets')
        if BOROUGH:
            streets_cols = streets_cols.append('borough')
        
        streets_cols_query = _querify_list(streets_cols, prefix = 'l', indent=20)

        with db_connection as con:
            # Register results as temp table
            con.register('_temp_results', results)

            # Join with locations table - explicitly select columns to avoid array issues
            enriched = con.execute(f"""
                SELECT
                    {results_cols_query},
                    {streets_cols_query}
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

            # Join with media_embeddings to get both image paths
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
                LEFT JOIN {universe_name}.media_embeddings e_from
                    ON r.location_id = e_from.location_id
                    AND r.year_from = e_from.year
                LEFT JOIN {universe_name}.media_embeddings e_to
                    ON r.location_id = e_to.location_id
                    AND r.year_to = e_to.year
                LEFT JOIN {universe_name}.locations l
                    ON r.location_id = l.location_id
            """).df()

            return enriched
    except Exception as e:
        logger.error(f"Error enriching change results with images: {e}")
        return results
