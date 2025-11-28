"""Backend search logic using streettransformer."""

import logging
import pandas as pd

from ..utils.display import enrich_results_with_streets, enrich_change_results_with_images
from streettransformer import StateLocationQuery, StateTextQuery, ChangeLocationQuery, CLIPEncoder

logger = logging.getLogger(__name__)

def search_by_location(
    config,
    db,
    db_connection_func,
    location_id: int,
    year: int,
    target_year: int | None = None,
    limit: int = 20,
    use_faiss: bool = True,
    use_whitening: bool = True
) -> pd.DataFrame:
    """Search for similar locations (thin wrapper around StateLocationQuery).

    Args:
        config: StreetTransformer Config object
        db: EmbeddingDB instance
        db_connection_func: Database connection function
        location_id: Query location ID
        year: Year of query location
        target_year: Optional target year to search in
        limit: Number of results
        use_faiss: Use FAISS for search
        use_whitening: Use whitening reranking

    Returns:
        DataFrame with search results enriched with street names
    """
    try:
        # Create and execute query
        query = StateLocationQuery(
            location_id=location_id,
            year=year,
            target_year=target_year,
            config=config,
            db=db,
            db_connection_func=db_connection_func,
            limit=limit,
            use_faiss=use_faiss,
            use_whitening=use_whitening
        )

        results = query.execute()

        # Dashboard-specific enrichment: add street names
        results = enrich_results_with_streets(results, db_connection_func, config.universe_name)

        return results

    except Exception as e:
        logger.error(f"Location search error: {e}")
        raise


def search_by_text(
    config,
    db,
    db_connection_func,
    text_query: str,
    year: int | None = None,
    limit: int = 20,
    use_faiss: bool = False,
    use_whitening: bool = False,
    clip_encoder=None
) -> pd.DataFrame:
    """Search for images matching text description (thin wrapper around StateTextQuery).

    Args:
        config: StreetTransformer Config object
        db: EmbeddingDB instance
        db_connection_func: Database connection function
        text_query: Text description
        year: Optional year filter
        limit: Number of results
        use_faiss: Use FAISS for search
        use_whitening: Use whitening reranking
        clip_encoder: Optional CLIPEncoder instance (created if not provided)

    Returns:
        DataFrame with search results enriched with street names
    """
    try:
        # Create encoder if not provided (for backward compatibility)
        if clip_encoder is None:
            clip_encoder = CLIPEncoder()

        # Create and execute query
        query = StateTextQuery(
            text_query=text_query,
            year=year,
            config=config,
            db=db,
            db_connection_func=db_connection_func,
            limit=limit,
            use_faiss=use_faiss,
            use_whitening=use_whitening,
            clip_encoder=clip_encoder
        )

        results = query.execute()

        # Dashboard-specific enrichment: add street names
        results = enrich_results_with_streets(results, db_connection_func, config.universe_name)

        return results

    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise


def search_change_patterns(
    config,
    db,
    db_connection_func,
    location_id: int,
    year_from: int,
    year_to: int,
    limit: int = 20
) -> pd.DataFrame:
    """Search for locations with similar change patterns (thin wrapper around ChangeLocationQuery).

    Args:
        config: StreetTransformer Config object
        db: EmbeddingDB instance
        db_connection_func: Database connection function
        location_id: Reference location
        year_from: Starting year
        year_to: Ending year
        limit: Number of results

    Returns:
        DataFrame with search results enriched with image paths
    """
    try:
        # Create and execute query
        query = ChangeLocationQuery(
            location_id=location_id,
            start_year=year_from,
            end_year=year_to,
            config=config,
            db=db,
            db_connection_func=db_connection_func,
            limit=limit
        )

        results = query.execute()

        # Dashboard-specific enrichment: add image paths for both years
        results = enrich_change_results_with_images(results, db_connection_func, config.universe_name)

        return results

    except Exception as e:
        logger.error(f"Change search error: {e}")
        raise


def get_embedding_stats(db, config) -> dict:
    """Get embedding statistics.

    Args:
        db: EmbeddingDB instance
        config: Config instance

    Returns:
        Dictionary with statistics
    """
    try:
        count = db.get_embedding_count()
        years = db.get_years()

        return {
            'total_embeddings': count,
            'years': years,
            'year_count': len(years),
            'vector_dim': config.vector_dim,
            'universe_name': config.universe_name
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise
