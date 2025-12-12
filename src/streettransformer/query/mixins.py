"""Mixins for composing query functionality.

This module provides orthogonal mixins for building query classes:
- Temporal mixins: StateMixin (point in time), ChangeMixin (delta)
- Functional mixins: DatabaseMixin, SearchMethodMixin
"""

import os
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import logging
from dotenv import load_dotenv
from abc import abstractmethod

load_dotenv()

from ..config import STConfig
from ..db.database import get_connection
from ..db.embedding_db import EmbeddingDB
from ..query.clip_embedding import CLIPEncoder

from ..image_retrieval.vector_db import VectorDB

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL MIXINS (Orthogonal Axis 1: State vs Change)
# =============================================================================

class StateMixin:
    """Mixin for state-based queries (single point in time).

    Attributes:
        location_id: Location identifier
        year: Year of interest
        target_year: Optional year to search within (None = all years)
    """
    location_id: str
    year: int
    target_years: Optional[list[int]] = None

    def get_temporal_key(self) -> str:
        """Get string representation of temporal parameters."""
        if self.target_year:
            return f"loc_{self.location_id}_y{self.year}_target[{'_'.join(self.target_years)}]"
        return f"loc_{self.location_id}_y{self.year}"

    def get_query_embedding_filter(self) -> str:
        """Get SQL filter for query embedding."""
        return f"location_id = {self.location_id} AND year = {self.year}"


class ChangeMixin:
    """Mixin for change-based queries (delta between two time points).

    Attributes:
        location_id: Location identifier
        year_from: Beginning year
        year_to: Ending year
        sequential_only: Whether to only search sequential year pairs
    """
    location_id: str
    year_from: int
    year_to: int
    target_years: Optional[list[int]] = None
    sequential_only: bool = False

    def get_temporal_key(self) -> str:
        """Get string representation of temporal parameters."""
        seq = "_seq" if self.sequential_only else ""
        return f"loc_{self.location_id}_change{self.year_from}to{self.year_to}{seq}"

    def get_change_vector_filter(self) -> str:
        """Get SQL filter for change vector."""
        return (f"location_id = {self.location_id} AND "
                f"year_from = {self.year_from} AND year_to = {self.year_to}")


# =============================================================================
# FUNCTIONAL MIXINS (Orthogonal Axis 2: Database, Search)
# =============================================================================

class DatabaseMixin:
    """Mixin providing database connection and query execution.

    Attributes:
        config: Configuration object with database path
        db: EmbeddingDB instance for vector operations (DuckDB-based)
        vector_db: VectorDB instance for PostgreSQL pgvector operations
        db_connection_func: Factory function for creating connections
    """
    config: STConfig
    db: EmbeddingDB
    vector_db: VectorDB
    db_connection_func: Optional[Callable] = None

    def get_connection(self):
        """Get database connection."""
        if self.db_connection_func:
            return self.db_connection_func()
        return get_connection(self.config.database_path, read_only=True)

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        with self.get_connection() as con:
            return con.execute(sql).df()

    def get_universe_table(self, table_name: str) -> str:
        """Get fully qualified table name for universe."""
        return f"{self.config.universe_name}.{table_name}"


class SearchMethodMixin:
    """Mixin providing search method configuration.

    Attributes:
        limit: Maximum number of results
        remove_self: Whether to remove query location from results
        use_faiss: Whether to use FAISS for approximate search
        use_whitening: Whether to apply whitening reranking
        artifacts_dir: Path to artifacts directory
        media_types: List of media types (e.g., ['image'], ['fusion'], ['mask'])
    """
    limit: int = 10
    remove_self: bool = True
    use_faiss: bool = False
    use_whitening: bool = True
    artifacts_dir: Path = Path(str(os.getenv('ARTIFACTS_DIR')))
    media_type: str = 'image'
    
    def __post_init__(self):
        if self.use_whitening or self.use_faiss:
            if not self.artifacts_dir:
                raise ValueError('Must set `artifacts_dir` or `ARTIFACTS_DIR` to use FAISS or Whitening')

    @property
    def media_suffix(self) -> str:
        """Get media type suffix for artifact filenames.

        Returns empty string for 'image', otherwise returns '_' + media_type.
        """
        match self.media_type:
            case "image":
                return ''
            case "sidebyside":
                return '_fusion'
            case 'mask':
                return '_mask'
            case _:
                raise ValueError(f"Unrecognized media_type '{self.media_type}'")

    @property
    def whitening_path(self) -> Optional[str]:
        """Get path to whitening file.

        Returns:
            Path to whiten.npz file in artifacts directory
        """
        if not self.artifacts_dir:
            return None
        
        match self.media_type:
            case 'image':
                return self.artifacts_dir / 'whiten.npz'
            case 'sidebyside':
                return self.artifacts_dir / 'fusion_whiten.npz'
            case 'mask': 
                return None
                # if self.should_whiten:
                #     raise ValueError(f'Unsure what to do with artifact_path {self}')
                # return None

    @abstractmethod
    @property
    def query_type_prefix(self) -> str:
        """Get query type prefix ('state' or 'delta').

        Override in subclasses to specify query type.
        """
        pass

    @property
    def faiss_index_path(self) -> Path:
        """Get path to FAISS index for this query type and media type.

        Returns:
            Path like: state_hnsw.faiss, delta_fusion_hnsw.faiss, etc.
        """
        if not self.artifacts_dir:
            logger.warning('No `artifacts_dir` found while trying to use faiss!')
            return None
        
        filename = f"{self.query_type_prefix}{self.media_suffix}_hnsw.faiss"
        faiss_path = self.artifacts_dir / filename
        if not faiss_path.exists():
            raise f"faiss_path '{faiss_path}' NOT found!"
            
        return faiss_path

    @property
    def search_method_name(self) -> str:
        """Get human-readable search method description."""
        method = "PostgreSQL pgvector"
        if self.use_faiss:
            method = "FAISS + " + method
        if self.use_whitening:
            method += " + Whitening"
        return method


class TextQueryMixin:
    text_query: str
    clip_encoder: CLIPEncoder = None
    target_years: Optional[list[str]] = None

    def __post_init__self(self):
        if self.clip_encoder is None:
            self.clip_encoder = CLIPEncoder()

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