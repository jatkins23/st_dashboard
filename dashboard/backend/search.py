"""Backend search logic using streettransformer."""

import logging
import numpy as np
import pandas as pd

from ..utils.encoding import encode_text_query
from ..utils.enrichment import enrich_results_with_streets, enrich_change_results_with_images

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
    """Search for similar locations.

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
        DataFrame with search results
    """
    try:
        # Get embedding for query location
        with db_connection_func() as con:
            query_df = con.execute(f"""
                SELECT location_id, location_key, year, image_path, embedding
                FROM {config.universe_name}.image_embeddings
                WHERE location_id = {location_id}
                    AND year = {year}
                    AND embedding IS NOT NULL
            """).df()

        if query_df.empty:
            logger.error(f"No embedding found for location {location_id} year {year}")
            return pd.DataFrame()

        query_vec = np.array(query_df.iloc[0]['embedding'])

        # Search using FAISS if requested
        if use_faiss:
            from streettransformer import FAISSIndexer
            indexer = FAISSIndexer(config)
            results = indexer.search(
                query_vector=query_vec,
                k=limit + 1,
                year=target_year if target_year else year
            )
        else:
            # Use database search
            results = db.search_similar(
                query_vector=query_vec,
                limit=limit + 1,
                year=target_year
            )

        # Remove self from results
        results = results[results['location_id'] != location_id]
        results = results.head(limit)

        # Apply whitening reranking if requested
        if use_whitening and not results.empty:
            from streettransformer import WhiteningTransform
            whiten = WhiteningTransform(config)
            results = whiten.rerank_results(
                query_vector=query_vec,
                results=results,
                year=target_year if target_year else year,
                top_k=limit
            )

        # Enrich with street names
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
    limit: int = 20
) -> pd.DataFrame:
    """Search for images matching text description.

    Args:
        config: StreetTransformer Config object
        db: EmbeddingDB instance
        db_connection_func: Database connection function
        text_query: Text description
        year: Optional year filter
        limit: Number of results

    Returns:
        DataFrame with search results
    """
    try:
        # Encode text query
        logger.info(f"Encoding text query: '{text_query}'")
        query_embedding = encode_text_query(text_query)

        # Search database
        results = db.search_similar(
            query_vector=query_embedding,
            limit=limit,
            year=year
        )

        # Enrich with street names
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
    """Search for locations with similar change patterns.

    Args:
        config: StreetTransformer Config object
        db: EmbeddingDB instance
        db_connection_func: Database connection function
        location_id: Reference location
        year_from: Starting year
        year_to: Ending year
        limit: Number of results

    Returns:
        DataFrame with search results
    """
    try:
        # Get embeddings for reference location
        with db_connection_func() as con:
            location_df = con.execute(f"""
                SELECT location_id, location_key, year, embedding
                FROM {config.universe_name}.image_embeddings
                WHERE location_id = {location_id}
                    AND year IN ({year_from}, {year_to})
                    AND embedding IS NOT NULL
            """).df()

        emb_from = location_df[location_df['year'] == year_from]
        emb_to = location_df[location_df['year'] == year_to]

        if emb_from.empty or emb_to.empty:
            logger.error(f"Missing embeddings for location {location_id}")
            return pd.DataFrame()

        # Compute query delta
        vec_from = np.array(emb_from.iloc[0]['embedding'])
        vec_to = np.array(emb_to.iloc[0]['embedding'])
        query_delta = vec_to - vec_from

        # Normalize
        norm = np.linalg.norm(query_delta)
        if norm > 0:
            query_delta = query_delta / norm

        # Search for similar changes
        results = db.search_change_vectors(
            query_delta=query_delta,
            limit=limit + 1,
            year_from=year_from,
            year_to=year_to
        )

        # Remove self
        results = results[results['location_id'] != location_id]
        results = results.head(limit)

        # Enrich with image paths for both years
        results = enrich_change_results_with_images(results, db_connection_func, config.universe_name)

        return results

    except Exception as e:
        logger.error(f"Change search error: {e}")
        raise


def get_embedding_stats(db) -> dict:
    """Get embedding statistics.

    Args:
        db: EmbeddingDB instance

    Returns:
        Dictionary with statistics
    """
    try:
        count = db.get_embedding_count()
        years = db.get_years()

        return {
            'total_embeddings': count,
            'years': years,
            'year_count': len(years)
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise
