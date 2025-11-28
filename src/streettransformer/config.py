"""Configuration for StreetTransformer embedding system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for StreetTransformer.

    Attributes:
        database_path: Path to DuckDB database (local or remote).
                      Supports local paths, HTTP/HTTPS URLs, S3, GCS, Azure, etc.
                      If not provided, reads from ST_DATABASE_PATH environment variable.
        universe_name: Name of the universe/schema to use
        vector_dim: Dimensionality of embeddings (default: 512 for CLIP)
        cache_dir: Directory for NPZ embedding caches
        index_dir: Directory for FAISS indexes
        stats_dir: Directory for whitening statistics

    Environment Variables:
        ST_DATABASE_PATH: Default database path if database_path not provided

    Example:
        >>> # Local database
        >>> config = Config(
        ...     database_path="/path/to/data.db",
        ...     universe_name="lion"
        ... )
        >>> # Remote database (HTTP/HTTPS)
        >>> config = Config(
        ...     database_path="https://example.com/data.ddb",
        ...     universe_name="lion"
        ... )
        >>> # S3 database
        >>> config = Config(
        ...     database_path="s3://bucket/path/data.ddb",
        ...     universe_name="lion"
        ... )
        >>> # Or use environment variable
        >>> os.environ['ST_DATABASE_PATH'] = "https://example.com/data.ddb"
        >>> config = Config(universe_name="lion")
    """

    database_path: str | Path | None = None
    universe_name: str = ""
    vector_dim: int = 512
    cache_dir: str | Path = "./data/embedding_cache"
    index_dir: str | Path = "./data/faiss_indexes"
    stats_dir: str | Path = "./data/whitening_stats"

    def __post_init__(self):
        """Convert paths to Path objects and create directories."""
        # Read database_path from environment if not provided
        if self.database_path is None:
            env_db_path = os.getenv('ST_DATABASE_PATH')
            if env_db_path is None:
                raise ValueError(
                    "database_path must be provided either as argument or via "
                    "ST_DATABASE_PATH environment variable"
                )
            self.database_path = env_db_path

        # Keep database_path as string if it's a URL or remote path
        # DuckDB supports: http://, https://, s3://, gs://, az://, etc.
        db_path_str = str(self.database_path)
        if any(db_path_str.startswith(prefix) for prefix in ['http://', 'https://', 's3://', 'gs://', 'az://', 'r2://']):
            # Remote path - keep as string for DuckDB
            self.database_path = db_path_str
        else:
            # Local path - convert to Path object
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
