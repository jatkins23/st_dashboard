"""Configuration for StreetTransformer embedding system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for StreetTransformer.

    Attributes:
        database_path: Path to DuckDB database (shared with st_preprocessing)
        universe_name: Name of the universe/schema to use
        vector_dim: Dimensionality of embeddings (default: 512 for CLIP)
        cache_dir: Directory for NPZ embedding caches
        index_dir: Directory for FAISS indexes
        stats_dir: Directory for whitening statistics

    Example:
        >>> config = Config(
        ...     database_path="/path/to/data.db",
        ...     universe_name="lion"
        ... )
        >>> config = Config.from_st_preprocessing("/path/to/st_preprocessing/data.db", "lion")
    """

    database_path: str | Path
    universe_name: str
    vector_dim: int = 512
    cache_dir: str | Path = "./data/embedding_cache"
    index_dir: str | Path = "./data/faiss_indexes"
    stats_dir: str | Path = "./data/whitening_stats"

    def __post_init__(self):
        """Convert paths to Path objects and create directories."""
        self.database_path = Path(self.database_path)
        self.cache_dir = Path(self.cache_dir)
        self.index_dir = Path(self.index_dir)
        self.stats_dir = Path(self.stats_dir)

        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_st_preprocessing(
        cls,
        database_path: str | Path,
        universe_name: str,
        **kwargs
    ) -> Config:
        """Create config that points to st_preprocessing database.

        Args:
            database_path: Path to st_preprocessing DuckDB database
            universe_name: Universe name (e.g., 'lion', 'nyc')
            **kwargs: Additional config parameters

        Returns:
            Config instance

        Example:
            >>> config = Config.from_st_preprocessing(
            ...     "/Users/jon/code/st_preprocessing/data.db",
            ...     "lion"
            ... )
        """
        return cls(
            database_path=database_path,
            universe_name=universe_name,
            **kwargs
        )
