"""Mixins for composing query functionality.

This module provides orthogonal mixins for building query classes:
- Temporal mixins: StateMixin (point in time), ChangeMixin (delta)
- Functional mixins: DatabaseMixin, SearchMethodMixin
"""

from typing import Optional, Callable
import pandas as pd
import logging

from ..config import STConfig
from ..db.database import get_connection
from ..db.embedding_db import EmbeddingDB
from ..query.clip_embedding import CLIPEncoder

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
        db: EmbeddingDB instance for vector operations
        db_connection_func: Factory function for creating connections
    """
    config: STConfig
    db: EmbeddingDB
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
    """Mixin providing different search method strategies.

    Attributes:
        use_faiss: Whether to use FAISS for approximate search
        use_whitening: Whether to apply whitening reranking
        limit: Maximum number of results
    """
    use_faiss: bool = True
    use_whitening: bool = False
    limit: int = 10
    remove_self: bool = True

    def get_search_method_name(self) -> str:
        """Get human-readable search method description."""
        methods = []
        if self.use_faiss:
            methods.append("FAISS")
        else:
            methods.append("Database")
        if self.use_whitening:
            methods.append("Whitening")
        return " + ".join(methods)


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