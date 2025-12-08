"""DuckDB-based vector embedding storage for StreetTransformer.

Adapted from the PostgreSQL/pgvector implementation for DuckDB compatibility.
"""

from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass
import logging
import json

import numpy as np
import pandas as pd

from .database import get_connection
from ..config import STConfig

logger = logging.getLogger(__name__)


@dataclass
class MediaEmbedding:
    """Container for media embedding data.

    Attributes:
        location_id: Unique location identifier
        location_key: Location key/name
        year: Year of the media
        media_type: Type of media - 'image', 'mask', or 'sidebyside'
        path: Path to the media file
        embedding: Embedding vector
        stats: Optional statistics (e.g., mask statistics)
    """
    location_id: int
    location_key: str
    year: int
    media_type: str
    path: str
    embedding: np.ndarray
    stats: dict[str, Any] | None = None


# Backward compatibility alias
ImageEmbedding = MediaEmbedding


class EmbeddingDB:
    """DuckDB-based vector embedding storage.

    This class provides functionality similar to PostgreSQL/pgvector
    but uses DuckDB with the VSS extension for vector similarity search.

    Key features:
    - Uses DuckDB VSS extension for vector similarity
    - Embeddings stored as FLOAT[] arrays
    - HNSW index support for fast similarity search
    - Cosine similarity via array operations

    Attributes:
        config: Configuration with database path and settings
        universe_name: Name of the universe (schema)
        vector_dim: Dimensionality of embeddings

    Example:
        >>> config = STConfig(database_path="data.db", universe_name="lion")
        >>> db = EmbeddingDB(config)
        >>> db.setup_schema()
        >>>
        >>> # Insert embeddings
        >>> embeddings = [
        ...     ImageEmbedding(
        ...         location_id=123,
        ...         location_key='street_123',
        ...         year=2020,
        ...         image_path='/path/to/image.png',
        ...         embedding=np.random.rand(512)
        ...     )
        ... ]
        >>> db.insert_embeddings(embeddings)
        >>>
        >>> # Search similar images
        >>> query_vec = np.random.rand(512)
        >>> results = db.search_similar(query_vec, limit=10)
    """

    def __init__(self, config: STConfig):
        """Initialize EmbeddingDB.

        Args:
            config: Configuration with database path and universe name
        """
        self.config = config
        self.universe_name = config.universe_name
        self.vector_dim = config.vector_dim
        self.schema = config.universe_name

    def setup_schema(
        self,
        create_index: bool = True,
        index_type: Literal['hnsw'] = 'hnsw',
        drop_existing: bool = False
    ) -> None:
        """Create schema and tables for embedding storage.

        Args:
            create_index: Whether to create VSS index
            index_type: Index type (currently only 'hnsw' supported)
            drop_existing: If True, drop existing tables before recreating
        """
        with get_connection(self.config.database_path) as con:
            # Create schema
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

            # Drop existing tables if requested
            if drop_existing:
                logger.info(f"Dropping existing tables in schema '{self.schema}'")
                con.execute(f"DROP TABLE IF EXISTS {self.schema}.media_embeddings CASCADE")
                con.execute(f"DROP TABLE IF EXISTS {self.schema}.change_vectors CASCADE")

            # Create media_embeddings table
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.media_embeddings (
                    location_id BIGINT NOT NULL,
                    location_key VARCHAR NOT NULL,
                    year INTEGER NOT NULL,
                    media_type VARCHAR NOT NULL,
                    path VARCHAR NOT NULL,
                    embedding FLOAT[{self.vector_dim}],
                    stats JSON,
                    PRIMARY KEY (location_key, year, media_type),
                    UNIQUE (path)
                )
            """)

            # Create change_vectors table for pre-computed deltas
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.change_vectors (
                    location_id BIGINT NOT NULL,
                    location_key VARCHAR NOT NULL,
                    media_type VARCHAR NOT NULL,
                    year_from INTEGER NOT NULL,
                    path_from VARCHAR NOT NULL,
                    year_to INTEGER NOT NULL,
                    path_to VARCHAR NOT NULL,
                    delta FLOAT[{self.vector_dim}],
                    PRIMARY KEY (location_key, year_from, year_to, media_type),
                    UNIQUE (path_from, path_to)
                )
            """)

            logger.info(f"Created embedding tables in schema '{self.schema}'")

            # Create VSS index for similarity search
            if create_index and index_type == 'hnsw':
                try:
                    # Create HNSW index on embedding column
                    con.execute(f"""
                        CREATE INDEX IF NOT EXISTS media_embeddings_hnsw_idx
                        ON {self.schema}.media_embeddings
                        USING HNSW (embedding)
                    """)
                    logger.info("Created HNSW index on embeddings")
                except Exception as e:
                    logger.warning(
                        f"Could not create HNSW index: {e}\n"
                        "HNSW indexes provide faster similarity search but require experimental features.\n"
                        "Similarity search will still work but may be slower for large datasets."
                    )

    def insert_embeddings(
        self,
        embeddings: list[MediaEmbedding],
        on_conflict: Literal['replace', 'ignore'] = 'replace'
    ) -> None:
        """Insert or update embeddings in the database.

        Args:
            embeddings: List of MediaEmbedding objects
            on_conflict: How to handle conflicts ('replace' or 'ignore')
        """
        if not embeddings:
            logger.warning("No embeddings to insert")
            return

        # Convert to DataFrame for bulk insert
        records = []
        for emb in embeddings:
            records.append({
                'location_id': emb.location_id,
                'location_key': emb.location_key,
                'year': emb.year,
                'media_type': emb.media_type,
                'path': emb.path,
                'embedding': emb.embedding.tolist(),
                'stats': json.dumps(emb.stats) if emb.stats else None
            })

        df = pd.DataFrame(records)

        with get_connection(self.config.database_path) as con:
            con.register('_tmp_embeddings', df)

            try:
                if on_conflict == 'replace':
                    # Upsert: insert or replace on conflict
                    con.execute(f"""
                        INSERT INTO {self.schema}.media_embeddings
                        SELECT * FROM _tmp_embeddings
                        ON CONFLICT (location_key, year, media_type)
                        DO UPDATE SET
                            location_id = excluded.location_id,
                            path = excluded.path,
                            embedding = excluded.embedding,
                            stats = excluded.stats
                    """)
                else:  # ignore
                    # Insert only if not exists
                    con.execute(f"""
                        INSERT INTO {self.schema}.media_embeddings
                        SELECT * FROM _tmp_embeddings
                        ON CONFLICT (location_key, year, media_type) DO NOTHING
                    """)

                logger.debug(f"Inserted {len(embeddings)} embeddings into {self.schema}.media_embeddings")
            finally:
                con.unregister('_tmp_embeddings')
    def get_stats(self) -> dict:
        """Get embedding statistics

        Returns:
            Dictionary with total embeddings, years, and year_count
        """
        return {
            'total_embeddings': self.get_embedding_count(),
            'years': self.get_years(),
            'year_count': len(self.get_years())
        }

    def search_similar(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        year: int | None = None,
        media_type: str | None = None
    ) -> pd.DataFrame:
        """Search for similar embeddings using cosine similarity.

        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            year: Optional year filter
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            DataFrame with columns: location_id, location_key, year, media_type, path, similarity
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            # Register query vector
            query_list = query_vector.tolist()

            # Build filters
            filters = ["embedding IS NOT NULL"]
            if year is not None:
                filters.append(f"year = {year}")
            if media_type is not None:
                filters.append(f"media_type = '{media_type}'")

            where_clause = " AND ".join(filters)

            # Cosine similarity using array operations
            query = f"""
                SELECT
                    location_id,
                    location_key,
                    year,
                    media_type,
                    path,
                    array_cosine_similarity(embedding, {query_list}::FLOAT[{self.vector_dim}]) AS similarity
                FROM {self.schema}.media_embeddings
                WHERE {where_clause}
                ORDER BY similarity DESC
                LIMIT {limit}
            """

            result_df = con.execute(query).df()
            return result_df

    def fetch_embeddings_by_location(
        self,
        location_key: str,
        media_type: str | None = None
    ) -> pd.DataFrame:
        """Fetch all embeddings for a given location across years.

        Args:
            location_key: Location identifier
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            DataFrame with years and embeddings
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            media_filter = f"AND media_type = '{media_type}'" if media_type else ""

            result_df = con.execute(f"""
                SELECT
                    location_id,
                    location_key,
                    year,
                    media_type,
                    path,
                    embedding
                FROM {self.schema}.media_embeddings
                WHERE location_key = '{location_key}'
                AND embedding IS NOT NULL
                {media_filter}
                ORDER BY year, media_type
            """).df()

            return result_df

    def fetch_embeddings_by_year(
        self,
        year: int,
        media_type: str | None = None
    ) -> pd.DataFrame:
        """Fetch all embeddings for a given year.

        Args:
            year: Year to fetch
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            DataFrame with locations and embeddings
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            media_filter = f"AND media_type = '{media_type}'" if media_type else ""

            result_df = con.execute(f"""
                SELECT
                    location_id,
                    location_key,
                    year,
                    media_type,
                    path,
                    embedding
                FROM {self.schema}.media_embeddings
                WHERE year = {year}
                AND embedding IS NOT NULL
                {media_filter}
                ORDER BY location_id, media_type
            """).df()

            return result_df

    def compute_change_vectors(
        self,
        year_from: int,
        year_to: int,
        media_type: str = 'image',
        normalize: bool = True
    ) -> None:
        """Compute and store change vectors (deltas) between two years.

        Args:
            year_from: Starting year
            year_to: Ending year
            media_type: Media type to compute changes for ('image', 'mask', 'sidebyside')
            normalize: Whether to normalize the delta vectors
        """
        # Fetch embeddings for both years
        from_df = self.fetch_embeddings_by_year(year_from, media_type=media_type)
        to_df = self.fetch_embeddings_by_year(year_to, media_type=media_type)

        # Merge on location_key to find matching locations
        merged = from_df.merge(
            to_df,
            on='location_key',
            suffixes=('_from', '_to')
        )

        if merged.empty:
            logger.warning(f"No matching locations between {year_from} and {year_to} for {media_type}")
            return

        # Compute deltas using vectorized operations
        embeddings_from = np.stack(merged['embedding_from'].values)
        embeddings_to = np.stack(merged['embedding_to'].values)

        deltas = embeddings_to - embeddings_from

        if normalize:
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            deltas = np.divide(deltas, norms, where=norms > 0)

        # Create records
        change_records = [
            {
                'location_id': row.location_id_from,
                'location_key': row.location_key,
                'media_type': media_type,
                'year_from': year_from,
                'path_from': row.path_from,
                'year_to': year_to,
                'path_to': row.path_to,
                'delta': delta.tolist()
            }
            for row, delta in zip(merged.itertuples(), deltas)
        ]

        if not change_records:
            logger.warning(f"No change vectors computed for {year_from} -> {year_to} ({media_type})")
            return

        # Insert change vectors
        change_df = pd.DataFrame(change_records)

        with get_connection(self.config.database_path) as con:
            con.register('_tmp_changes', change_df)

            try:
                con.execute(f"""
                    INSERT INTO {self.schema}.change_vectors
                    SELECT * FROM _tmp_changes
                    ON CONFLICT (location_key, year_from, year_to, media_type)
                    DO UPDATE SET
                        location_id = excluded.location_id,
                        path_from = excluded.path_from,
                        path_to = excluded.path_to,
                        delta = excluded.delta
                """)

                logger.info(
                    f"Computed {len(change_records)} change vectors for "
                    f"{year_from} -> {year_to} ({media_type})"
                )
            finally:
                con.unregister('_tmp_changes')

    def search_change_vectors(
        self,
        query_delta: np.ndarray,
        limit: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
        media_type: str | None = None
    ) -> pd.DataFrame:
        """Search for similar change patterns.

        Args:
            query_delta: Query change vector
            limit: Number of results
            year_from: Optional starting year filter
            year_to: Optional ending year filter
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            DataFrame with matching change vectors and similarity scores
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            query_list = query_delta.tolist()

            # Build filters
            filters = []
            if year_from is not None:
                filters.append(f"year_from = {year_from}")
            if year_to is not None:
                filters.append(f"year_to = {year_to}")
            if media_type is not None:
                filters.append(f"media_type = '{media_type}'")

            where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

            query = f"""
                SELECT
                    location_id,
                    location_key,
                    media_type,
                    year_from,
                    path_from,
                    year_to,
                    path_to,
                    array_cosine_similarity(delta, {query_list}::FLOAT[{self.vector_dim}]) AS similarity
                FROM {self.schema}.change_vectors
                {where_clause}
                ORDER BY similarity DESC
                LIMIT {limit}
            """

            result_df = con.execute(query).df()
            return result_df

    def get_embedding_count(self, media_type: str | None = None) -> int:
        """Get total number of embeddings in database.

        Args:
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            Total count of embeddings
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            media_filter = f"WHERE media_type = '{media_type}'" if media_type else ""
            count = con.execute(f"""
                SELECT COUNT(*) as cnt
                FROM {self.schema}.media_embeddings
                {media_filter}
            """).fetchone()[0]
            return count

    def get_years(self, media_type: str | None = None) -> list[int]:
        """Get all years with embeddings.

        Args:
            media_type: Optional media type filter ('image', 'mask', 'sidebyside')

        Returns:
            Sorted list of years
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            media_filter = f"WHERE media_type = '{media_type}'" if media_type else ""
            years = con.execute(f"""
                SELECT DISTINCT year
                FROM {self.schema}.media_embeddings
                {media_filter}
                ORDER BY year
            """).df()['year'].tolist()
            return years

    def get_media_types(self) -> list[str]:
        """Get all media types with embeddings.

        Returns:
            Sorted list of media types
        """
        with get_connection(self.config.database_path, read_only=True) as con:
            media_types = con.execute(f"""
                SELECT DISTINCT media_type
                FROM {self.schema}.media_embeddings
                ORDER BY media_type
            """).df()['media_type'].tolist()
            return media_types
