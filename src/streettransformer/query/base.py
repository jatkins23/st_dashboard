"""Base query classes for streettransformer.

This module provides the abstract base class that all queries inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd

from ..config import Config
from ..db.embedding_db import EmbeddingDB
from .mixins import DatabaseMixin, SearchMethodMixin


@dataclass
class BaseQuery(ABC, DatabaseMixin, SearchMethodMixin):
    """Abstract base for all queries.

    Provides common functionality and enforces interface.
    All concrete query classes should inherit from this and implement
    the execute() and get_cache_key() methods.

    Attributes:
        config: StreetTransformer configuration (includes database_path and universe_name)
        db: EmbeddingDB instance for vector operations (created from config, passed for reuse)
        limit: Maximum number of results to return
        use_faiss: Whether to use FAISS for search
        use_whitening: Whether to apply whitening reranking
        db_connection_func: Optional database connection factory
    """
    config: Config
    db: EmbeddingDB
    limit: int = 10
    use_faiss: bool = True
    use_whitening: bool = False
    db_connection_func: Optional[Callable] = None

    @abstractmethod
    def execute(self) -> pd.DataFrame:
        """Execute query and return results.

        Returns:
            DataFrame with search results including similarity scores
        """
        pass

    @abstractmethod
    def get_cache_key(self) -> str:
        """Get unique cache key for this query.

        Returns:
            String cache key incorporating all query parameters
        """
        pass
