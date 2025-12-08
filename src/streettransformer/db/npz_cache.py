"""NPZ-based caching for embeddings.

This module provides efficient caching of embeddings using NumPy's compressed
.npz format. This is useful for:
- Fast loading of embeddings without database queries
- Offline processing and analysis
- Sharing embeddings across systems

The cache stores:
- Embeddings as float32 arrays
- Metadata (location_id, location_key, year, image_path)
- Cache creation timestamp and statistics

Usage:
    >>> config = STConfig(database_path="data.db", universe_name="lion")
    >>> cache = NPZCache(config)
    >>> cache.build_from_db(year=2020)
    >>>
    >>> # Load embeddings from cache
    >>> data = cache.load(year=2020)
    >>> embeddings = data['embeddings']
    >>> metadata = data['metadata']
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .database import get_connection
from ..config import STConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about a cached embedding file.

    Attributes:
        year: Year of cached embeddings
        num_embeddings: Number of embeddings in cache
        vector_dim: Dimensionality of embeddings
        file_size_mb: Size of cache file in megabytes
        created_at: ISO timestamp of cache creation
        file_path: Path to cache file
    """
    year: int
    num_embeddings: int
    vector_dim: int
    file_size_mb: float
    created_at: str
    file_path: Path


class NPZCache:
    """Manage NPZ-based embedding caches.

    Attributes:
        config: Configuration with database path and settings
        universe_name: Name of the universe (schema)
        cache_dir: Directory for cache files
        vector_dim: Embedding dimensionality
    """

    def __init__(self, config: STConfig):
        """Initialize NPZ cache manager.

        Args:
            config: Configuration with database path and cache settings
        """
        self.config = config
        self.universe_name = config.universe_name
        self.cache_dir = config.cache_dir
        self.vector_dim = config.vector_dim

    def _get_cache_path(self, year: int, embedding_type: str = 'embedding') -> Path:
        """Get path to cache file for a specific year.

        Args:
            year: Year
            embedding_type: Type of embedding (embedding, mask_embedding, etc.)

        Returns:
            Path to cache file
        """
        filename = f"{self.universe_name}_{embedding_type}_{year}.npz"
        return self.cache_dir / filename

    def build_from_db(
        self,
        year: int | None = None,
        embedding_type: str = 'embedding',
        force_rebuild: bool = False
    ) -> dict[int, Path]:
        """Build NPZ cache from database.

        Args:
            year: Specific year to cache (None for all years)
            embedding_type: Type of embedding to cache
            force_rebuild: Force rebuild even if cache exists

        Returns:
            Dictionary mapping year to cache file path
        """
        # Get available years if not specified
        with get_connection(self.config.database_path, read_only=True) as con:
            if year is None:
                years_df = con.execute(f"""
                    SELECT DISTINCT year
                    FROM {self.universe_name}.media_embeddings
                    WHERE {embedding_type} IS NOT NULL
                    ORDER BY year
                """).df()
                years = years_df['year'].tolist()
            else:
                years = [year]

        cache_paths = {}

        for yr in years:
            cache_path = self._get_cache_path(yr, embedding_type)

            # Skip if exists and not forcing rebuild
            if cache_path.exists() and not force_rebuild:
                logger.info(f"Cache already exists for year {yr}: {cache_path}")
                cache_paths[yr] = cache_path
                continue

            logger.info(f"Building cache for year {yr}...")

            # Load embeddings from database
            with get_connection(self.config.database_path, read_only=True) as con:
                df = con.execute(f"""
                    SELECT
                        location_id,
                        location_key,
                        year,
                        media_type,
                        path,
                        {embedding_type}
                    FROM {self.universe_name}.media_embeddings
                    WHERE year = {yr}
                        AND {embedding_type} IS NOT NULL
                    ORDER BY location_id, media_type
                """).df()

            if df.empty:
                logger.warning(f"No embeddings found for year {yr}")
                continue

            # Extract embeddings and metadata
            embeddings = np.stack(df[embedding_type].values).astype(np.float32)

            metadata = {
                'location_ids': df['location_id'].values,
                'location_keys': df['location_key'].values,
                'years': df['year'].values,
                'media_types': df['media_type'].values,
                'paths': df['path'].values,
                'created_at': datetime.now().isoformat(),
                'universe_name': self.universe_name,
                'embedding_type': embedding_type,
                'vector_dim': embeddings.shape[1],
                'num_embeddings': len(embeddings)
            }

            # Save to NPZ
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                **metadata
            )

            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Cached {len(embeddings)} embeddings for year {yr} "
                f"({file_size_mb:.2f} MB): {cache_path}"
            )

            cache_paths[yr] = cache_path

        return cache_paths

    def load(
        self,
        year: int,
        embedding_type: str = 'embedding'
    ) -> dict[str, Any]:
        """Load embeddings from cache.

        Args:
            year: Year to load
            embedding_type: Type of embedding

        Returns:
            Dictionary with keys:
                - embeddings: np.ndarray of shape (n, vector_dim)
                - metadata: DataFrame with location_id, location_key, year, image_path
                - info: Cache metadata

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = self._get_cache_path(year, embedding_type)

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found for year {year}: {cache_path}\n"
                f"Run build_from_db() first to create the cache."
            )

        # Load NPZ file
        data = np.load(cache_path, allow_pickle=True)

        # Extract embeddings
        embeddings = data['embeddings']

        # Build metadata DataFrame
        metadata = pd.DataFrame({
            'location_id': data['location_ids'],
            'location_key': data['location_keys'].astype(str),
            'year': data['years'],
            'media_type': data['media_types'].astype(str),
            'path': data['paths'].astype(str)
        })

        # Extract cache info
        info = {
            'created_at': str(data['created_at']),
            'universe_name': str(data['universe_name']),
            'embedding_type': str(data['embedding_type']),
            'vector_dim': int(data['vector_dim']),
            'num_embeddings': int(data['num_embeddings'])
        }

        logger.info(f"Loaded {len(embeddings)} embeddings from cache: {cache_path}")

        return {
            'embeddings': embeddings,
            'metadata': metadata,
            'info': info
        }

    def get_cache_info(self, year: int, embedding_type: str = 'embedding') -> CacheInfo | None:
        """Get information about a cached file.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            CacheInfo object or None if cache doesn't exist
        """
        cache_path = self._get_cache_path(year, embedding_type)

        if not cache_path.exists():
            return None

        # Load metadata only (without loading full arrays)
        data = np.load(cache_path, allow_pickle=True)

        return CacheInfo(
            year=year,
            num_embeddings=int(data['num_embeddings']),
            vector_dim=int(data['vector_dim']),
            file_size_mb=cache_path.stat().st_size / (1024 * 1024),
            created_at=str(data['created_at']),
            file_path=cache_path
        )

    def list_caches(self) -> list[CacheInfo]:
        """List all available caches.

        Returns:
            List of CacheInfo objects sorted by year
        """
        caches = []

        # Find all NPZ files matching pattern
        pattern = f"{self.universe_name}_*.npz"
        for cache_file in self.cache_dir.glob(pattern):
            try:
                # Parse filename to extract year and type
                # Format: {universe}_{type}_{year}.npz
                parts = cache_file.stem.split('_')
                if len(parts) >= 3:
                    year = int(parts[-1])
                    embedding_type = '_'.join(parts[1:-1])

                    info = self.get_cache_info(year, embedding_type)
                    if info:
                        caches.append(info)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse cache file: {cache_file}: {e}")

        return sorted(caches, key=lambda x: x.year)

    def delete_cache(self, year: int, embedding_type: str = 'embedding') -> bool:
        """Delete a cache file.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            True if deleted, False if didn't exist
        """
        cache_path = self._get_cache_path(year, embedding_type)

        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted cache: {cache_path}")
            return True

        return False

    def validate_cache(self, year: int, embedding_type: str = 'embedding') -> bool:
        """Validate cache against database.

        Checks if:
        - Cache file exists
        - Number of embeddings matches database
        - Embedding dimensions are correct

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            True if cache is valid
        """
        cache_path = self._get_cache_path(year, embedding_type)

        if not cache_path.exists():
            logger.warning(f"Cache file does not exist: {cache_path}")
            return False

        # Load cache info
        try:
            data = self.load(year, embedding_type)
            cache_count = len(data['embeddings'])
            cache_dim = data['embeddings'].shape[1]
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

        # Check against database
        with get_connection(self.config.database_path, read_only=True) as con:
            db_count = con.execute(f"""
                SELECT COUNT(*) as count
                FROM {self.universe_name}.media_embeddings
                WHERE year = {year}
                    AND {embedding_type} IS NOT NULL
            """).fetchone()[0]

        if cache_count != db_count:
            logger.warning(
                f"Cache count mismatch: cache={cache_count}, db={db_count}"
            )
            return False

        if cache_dim != self.vector_dim:
            logger.warning(
                f"Dimension mismatch: cache={cache_dim}, expected={self.vector_dim}"
            )
            return False

        logger.info(f"Cache validated successfully for year {year}")
        return True
