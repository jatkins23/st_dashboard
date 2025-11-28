"""Concrete query classes for different search types.

This module provides the main query classes for:
- StateLocationQuery: Search for similar locations at a point in time
- ChangeLocationQuery: Search for similar change patterns
- StateTextQuery: Search for locations matching text description

TODO: Refactor to remove db_connection_func redundancy
    Currently we pass both `db` (EmbeddingDB) and `db_connection_func` (raw connection factory).
    This is redundant - EmbeddingDB could expose methods for getting embeddings directly.
    Consider adding methods to EmbeddingDB and removing db_connection_func parameter.
"""

from typing import Optional
import pandas as pd
import hashlib

from .base import BaseQuery
from .mixins import StateMixin, ChangeMixin


class StateLocationQuery(BaseQuery, StateMixin):
    """Query for similar locations at a point in time.

    This query finds locations with similar visual appearance to a reference
    location at a specific year. Optionally can search within a target year.

    Example:
        >>> from streettransformer import Config, EmbeddingDB
        >>> config = Config(database_path="data.db", universe_name="lion")
        >>> db = EmbeddingDB(config)
        >>> query = StateLocationQuery(
        ...     location_id=123,
        ...     year=2020,
        ...     config=config,
        ...     db=db
        ... )
        >>> results = query.execute()
    """

    def __init__(
        self,
        location_id: int,
        year: int,
        config,
        db,
        target_year: Optional[int] = None,
        limit: int = 10,
        use_faiss: bool = True,
        use_whitening: bool = False,
        db_connection_func = None
    ):
        """Initialize StateLocationQuery.

        Args:
            location_id: Location identifier
            year: Year of interest
            config: StreetTransformer configuration
            db: EmbeddingDB instance
            target_year: Optional target year to search within
            limit: Maximum number of results
            use_faiss: Whether to use FAISS for search
            use_whitening: Whether to apply whitening reranking
            db_connection_func: Optional database connection factory
        """
        # StateMixin fields
        self.location_id = location_id
        self.year = year
        self.target_year = target_year

        # BaseQuery fields
        self.config = config
        self.db = db
        self.limit = limit
        self.use_faiss = use_faiss
        self.use_whitening = use_whitening
        self.db_connection_func = db_connection_func

    def execute(self) -> pd.DataFrame:
        """Execute state-based location similarity search.

        Returns:
            DataFrame with columns: location_id, year, similarity, image_path, etc.
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        # Get embedding for query location
        query_df = self.execute_query(f"""
            SELECT location_id, location_key, year, image_path, embedding
            FROM {self.get_universe_table('image_embeddings')}
            WHERE location_id = {self.location_id}
                AND year = {self.year}
                AND embedding IS NOT NULL
        """)

        if query_df.empty:
            logger.error(f"No embedding found for location {self.location_id} year {self.year}")
            return pd.DataFrame()

        query_vec = np.array(query_df.iloc[0]['embedding'])

        # Search using FAISS if requested
        if self.use_faiss:
            from ..faiss_index import FAISSIndexer
            indexer = FAISSIndexer(self.config)
            results = indexer.search(
                query_vector=query_vec,
                k=self.limit + 1,
                year=self.target_year if self.target_year else self.year
            )
        else:
            # Use database search
            results = self.db.search_similar(
                query_vector=query_vec,
                limit=self.limit + 1,
                year=self.target_year
            )

        # Remove self from results
        results = results[results['location_id'] != self.location_id]
        results = results.head(self.limit)

        # Apply whitening reranking if requested
        if self.use_whitening and not results.empty:
            from ..whitening import WhiteningTransform
            whiten = WhiteningTransform(self.config)
            results = whiten.rerank_results(
                query_vector=query_vec,
                results=results,
                year=self.target_year if self.target_year else self.year,
                top_k=self.limit
            )

        return results

    def get_cache_key(self) -> str:
        """Get unique cache key for this query.

        Returns:
            String like: loc_123_y2020_target2021_fw_n10
        """
        temporal = self.get_temporal_key()
        method = "f" if self.use_faiss else "d"
        method += "w" if self.use_whitening else ""
        return f"{temporal}_{method}_n{self.limit}"


class ChangeLocationQuery(BaseQuery, ChangeMixin):
    """Query for locations with similar change patterns.

    This query finds locations that experienced similar visual changes between
    two time periods, based on embedding delta vectors.

    Example:
        >>> query = ChangeLocationQuery(
        ...     location_id=123,
        ...     start_year=2015,
        ...     end_year=2020,
        ...     config=config,
        ...     db=db
        ... )
        >>> results = query.execute()
    """

    def __init__(
        self,
        location_id: int,
        start_year: int,
        end_year: int,
        config,
        db,
        sequential_only: bool = False,
        limit: int = 10,
        use_faiss: bool = True,
        use_whitening: bool = False,
        db_connection_func = None
    ):
        """Initialize ChangeLocationQuery.

        Args:
            location_id: Location identifier
            start_year: Starting year for change detection
            end_year: Ending year for change detection
            config: StreetTransformer configuration
            db: EmbeddingDB instance
            sequential_only: Whether to only consider sequential year pairs
            limit: Maximum number of results
            use_faiss: Whether to use FAISS for search
            use_whitening: Whether to apply whitening reranking
            db_connection_func: Optional database connection factory
        """
        # ChangeMixin fields
        self.location_id = location_id
        self.start_year = start_year
        self.end_year = end_year
        self.sequential_only = sequential_only

        # BaseQuery fields
        self.config = config
        self.db = db
        self.limit = limit
        self.use_faiss = use_faiss
        self.use_whitening = use_whitening
        self.db_connection_func = db_connection_func

    def execute(self) -> pd.DataFrame:
        """Execute change pattern search.

        Returns:
            DataFrame with columns: location_id, year_from, year_to, similarity, etc.
        """
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        # Get embeddings for reference location
        location_df = self.execute_query(f"""
            SELECT location_id, location_key, year, embedding
            FROM {self.get_universe_table('image_embeddings')}
            WHERE location_id = {self.location_id}
                AND year IN ({self.start_year}, {self.end_year})
                AND embedding IS NOT NULL
        """)

        emb_from = location_df[location_df['year'] == self.start_year]
        emb_to = location_df[location_df['year'] == self.end_year]

        if emb_from.empty or emb_to.empty:
            logger.error(f"Missing embeddings for location {self.location_id}")
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
        results = self.db.search_change_vectors(
            query_delta=query_delta,
            limit=self.limit + 1,
            year_from=self.start_year,
            year_to=self.end_year
        )

        # Remove self
        results = results[results['location_id'] != self.location_id]
        results = results.head(self.limit)

        return results

    def get_cache_key(self) -> str:
        """Get unique cache key for this query.

        Returns:
            String like: loc_123_change2015to2020_n10
        """
        temporal = self.get_temporal_key()
        return f"{temporal}_n{self.limit}"


class StateTextQuery(BaseQuery, StateMixin):
    """Query for locations matching text description at a point in time.

    Uses CLIP text encoding to find images that match a text description.
    This enables natural language queries like "street with trees and parked cars".

    Example:
        >>> from streettransformer import CLIPEncoder
        >>> encoder = CLIPEncoder()  # Create once at startup
        >>> query = StateTextQuery(
        ...     text_query="street with trees",
        ...     year=2020,
        ...     config=config,
        ...     db=db,
        ...     clip_encoder=encoder
        ... )
        >>> results = query.execute()
    """

    def __init__(
        self,
        text_query: str,
        config,
        db,
        clip_encoder,
        year: Optional[int] = None,
        limit: int = 10,
        use_faiss: bool = True,
        use_whitening: bool = False,
        db_connection_func = None
    ):
        """Initialize StateTextQuery.

        Args:
            text_query: Text description to search for
            config: StreetTransformer configuration
            db: EmbeddingDB instance
            clip_encoder: CLIPEncoder instance for text encoding
            year: Optional year filter (None = search all years)
            limit: Maximum number of results
            use_faiss: Whether to use FAISS for search
            use_whitening: Whether to apply whitening reranking
            db_connection_func: Optional database connection factory
        """
        # Text query specific fields
        self.text_query = text_query
        self.year = year
        self.clip_encoder = clip_encoder

        # BaseQuery fields
        self.config = config
        self.db = db
        self.limit = limit
        self.use_faiss = use_faiss
        self.use_whitening = use_whitening
        self.db_connection_func = db_connection_func

    def execute(self) -> pd.DataFrame:
        """Execute text-to-image search.

        Returns:
            DataFrame with columns: location_id, year, similarity, image_path, etc.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Encode text query
        if self.clip_encoder is None:
            raise ValueError(
                "clip_encoder is required for text queries. "
                "Create a CLIPEncoder instance and pass it to the query."
            )

        logger.info(f"Encoding text query: '{self.text_query}'")
        query_embedding = self.clip_encoder.encode(self.text_query)

        # Search using FAISS if requested
        if self.use_faiss:
            from ..faiss_index import FAISSIndexer
            indexer = FAISSIndexer(self.config)
            results = indexer.search(
                query_vector=query_embedding,
                k=self.limit,
                year=self.year
            )
        else:
            # Use database search
            results = self.db.search_similar(
                query_vector=query_embedding,
                limit=self.limit,
                year=self.year
            )

        # Apply whitening reranking if requested
        if self.use_whitening and not results.empty:
            from ..whitening import WhiteningTransform
            whiten = WhiteningTransform(self.config)
            results = whiten.rerank_results(
                query_vector=query_embedding,
                results=results,
                year=self.year if self.year else None,
                top_k=self.limit
            )

        return results

    def get_cache_key(self) -> str:
        """Get unique cache key for this query.

        Uses MD5 hash of text query to keep key manageable.

        Returns:
            String like: text_a3f2b8c1_y2020_fw_n10
        """
        text_hash = hashlib.md5(self.text_query.encode()).hexdigest()[:8]
        year_str = f"y{self.year}" if self.year else "all_years"
        method = "f" if self.use_faiss else "d"
        method += "w" if self.use_whitening else ""
        return f"text_{text_hash}_{year_str}_{method}_n{self.limit}"

    # Override StateMixin methods since text queries don't have location_id
    def get_temporal_key(self) -> str:
        """Get temporal key for text query.

        Returns:
            String like: text_y2020 or text_all_years
        """
        return f"text_y{self.year}" if self.year else "text_all_years"

    def get_query_embedding_filter(self) -> str:
        """Get SQL filter for query embedding.

        Returns:
            SQL WHERE clause fragment
        """
        if self.year:
            return f"year = {self.year}"
        return "1=1"  # No filter for all years
