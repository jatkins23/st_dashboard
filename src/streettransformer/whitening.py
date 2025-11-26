"""PCA whitening for embedding normalization and reranking.

This module implements PCA-based whitening transformation to improve
retrieval quality by removing dominant directions in the embedding space.

Whitening helps by:
- Normalizing the distribution of embeddings
- Reducing the influence of dominant features
- Improving similarity scores for retrieval

The typical workflow is:
1. Compute whitening statistics (PCA) on a sample of embeddings
2. Apply whitening transformation to query vectors
3. Rerank FAISS/database results using whitened similarities

Usage:
    >>> config = Config(database_path="data.db", universe_name="lion")
    >>> whiten = WhiteningTransform(config)
    >>> whiten.compute_statistics(year=2020, n_components=512)
    >>>
    >>> # Apply whitening to query
    >>> query_whitened = whiten.transform(query_embedding)
    >>>
    >>> # Rerank results
    >>> reranked = whiten.rerank_results(
    ...     query_vector=query_embedding,
    ...     results=faiss_results,
    ...     year=2020
    ... )

Reference:
    "Whitening and re-ranking for image search" (Jégou & Chum, 2012)
    https://hal.inria.fr/hal-00722622v1/document
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
from dataclasses import dataclass
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .npz_cache import NPZCache
from .config import Config
from .database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class WhiteningStats:
    """Whitening transformation statistics.

    Attributes:
        year: Year of embeddings used for statistics
        n_samples: Number of samples used for PCA
        n_components: Number of PCA components
        mean: Mean vector for centering
        components: PCA components matrix
        explained_variance: Variance explained by each component
        created_at: ISO timestamp of creation
    """
    year: int
    n_samples: int
    n_components: int
    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    created_at: str


class WhiteningTransform:
    """PCA-based whitening transformation.

    Attributes:
        config: Configuration with database path and settings
        universe_name: Name of universe
        stats_dir: Directory for whitening statistics
        cache: NPZ cache for loading embeddings
    """

    def __init__(self, config: Config):
        """Initialize whitening transform.

        Args:
            config: Configuration with database path and settings
        """
        self.config = config
        self.universe_name = config.universe_name
        self.stats_dir = config.stats_dir
        self.vector_dim = config.vector_dim
        self.cache = NPZCache(config)

        # Current loaded stats
        self._current_stats: WhiteningStats | None = None
        self._current_year: int | None = None

    def _get_stats_path(self, year: int, embedding_type: str = 'embedding') -> Path:
        """Get path to whitening statistics file.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            Path to pickle file
        """
        filename = f"{self.universe_name}_{embedding_type}_whitening_{year}.pkl"
        return self.stats_dir / filename

    def compute_statistics(
        self,
        year: int,
        embedding_type: str = 'embedding',
        n_components: int | None = None,
        n_samples: int | None = None,
        use_cache: bool = True,
        force_recompute: bool = False
    ) -> WhiteningStats:
        """Compute whitening statistics using PCA.

        Args:
            year: Year to compute statistics for
            embedding_type: Type of embedding
            n_components: Number of PCA components (default: vector_dim)
            n_samples: Number of samples to use (default: all)
            use_cache: Use NPZ cache if available
            force_recompute: Force recomputation even if stats exist

        Returns:
            WhiteningStats object
        """
        stats_path = self._get_stats_path(year, embedding_type)

        # Check if stats already exist
        if stats_path.exists() and not force_recompute:
            logger.info(f"Loading existing whitening stats: {stats_path}")
            return self.load_statistics(year, embedding_type)

        if n_components is None:
            n_components = self.vector_dim

        logger.info(f"Computing whitening statistics for year {year}...")

        # Load embeddings
        if use_cache:
            try:
                data = self.cache.load(year, embedding_type)
                embeddings = data['embeddings']
                logger.info(f"Loaded {len(embeddings)} embeddings from cache")
            except FileNotFoundError:
                logger.info("Cache not found, loading from database...")
                with get_connection(self.config.database_path, read_only=True) as con:
                    df = con.execute(f"""
                        SELECT {embedding_type}
                        FROM {self.universe_name}.image_embeddings
                        WHERE year = {year}
                            AND {embedding_type} IS NOT NULL
                    """).df()
                embeddings = np.stack(df[embedding_type].values).astype(np.float32)
        else:
            with get_connection(self.config.database_path, read_only=True) as con:
                df = con.execute(f"""
                    SELECT {embedding_type}
                    FROM {self.universe_name}.image_embeddings
                    WHERE year = {year}
                        AND {embedding_type} IS NOT NULL
                """).df()
            embeddings = np.stack(df[embedding_type].values).astype(np.float32)

        # Sample if requested
        if n_samples is not None and n_samples < len(embeddings):
            logger.info(f"Sampling {n_samples} embeddings for PCA")
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]

        # Compute PCA
        logger.info(f"Computing PCA with {n_components} components on {len(embeddings)} samples...")
        pca = PCA(n_components=n_components, whiten=False)
        pca.fit(embeddings)

        # Create stats object
        stats = WhiteningStats(
            year=year,
            n_samples=len(embeddings),
            n_components=n_components,
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            created_at=datetime.now().isoformat()
        )

        # Save statistics
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        logger.info(f"Saved whitening statistics to {stats_path}")
        logger.info(f"Explained variance (first 5 components): {stats.explained_variance[:5]}")

        # Load into current
        self._current_stats = stats
        self._current_year = year

        return stats

    def load_statistics(self, year: int, embedding_type: str = 'embedding') -> WhiteningStats:
        """Load whitening statistics from file.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            WhiteningStats object

        Raises:
            FileNotFoundError: If statistics don't exist
        """
        stats_path = self._get_stats_path(year, embedding_type)

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Whitening statistics not found: {stats_path}\n"
                f"Run compute_statistics() first."
            )

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self._current_stats = stats
        self._current_year = year

        logger.info(
            f"Loaded whitening statistics for year {year} "
            f"({stats.n_components} components, {stats.n_samples} samples)"
        )

        return stats

    def transform(
        self,
        vectors: np.ndarray,
        year: int | None = None,
        embedding_type: str = 'embedding'
    ) -> np.ndarray:
        """Apply whitening transformation to vectors.

        Args:
            vectors: Input vectors (n, dim) or (dim,)
            year: Year (loads stats if not already loaded)
            embedding_type: Type of embedding

        Returns:
            Whitened vectors (same shape as input)

        Raises:
            ValueError: If no statistics are loaded
        """
        # Handle single vector
        single_vector = False
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            single_vector = True

        # Load stats if needed
        if self._current_stats is None or (year is not None and self._current_year != year):
            if year is None:
                raise ValueError("Must specify year when no statistics are loaded")
            self.load_statistics(year, embedding_type)

        # Center vectors
        centered = vectors - self._current_stats.mean

        # Project onto PCA components and normalize by variance
        whitened = np.dot(centered, self._current_stats.components.T)
        whitened = whitened / np.sqrt(self._current_stats.explained_variance)

        # Normalize to unit length
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        whitened = whitened / (norms + 1e-8)

        if single_vector:
            whitened = whitened[0]

        return whitened.astype(np.float32)

    def rerank_results(
        self,
        query_vector: np.ndarray,
        results: pd.DataFrame,
        year: int,
        embedding_type: str = 'embedding',
        top_k: int | None = None
    ) -> pd.DataFrame:
        """Rerank search results using whitened similarities.

        This loads the embeddings for all results, applies whitening,
        and recomputes similarities with the whitened query.

        Args:
            query_vector: Original query embedding
            results: Results DataFrame (must have 'location_id' column)
            year: Year
            embedding_type: Type of embedding
            top_k: Return only top k results (default: all)

        Returns:
            Reranked DataFrame with updated similarity scores
        """
        if results.empty:
            return results

        # Load embeddings for all results
        location_ids = results['location_id'].tolist()
        location_ids_str = ','.join(map(str, location_ids))

        with get_connection(self.config.database_path, read_only=True) as con:
            embeddings_df = con.execute(f"""
                SELECT location_id, {embedding_type}
                FROM {self.universe_name}.image_embeddings
                WHERE location_id IN ({location_ids_str})
                    AND year = {year}
                    AND {embedding_type} IS NOT NULL
            """).df()

        if embeddings_df.empty:
            logger.warning("No embeddings found for reranking")
            return results

        # Extract embeddings
        result_embeddings = np.stack(embeddings_df[embedding_type].values).astype(np.float32)

        # Apply whitening to query and results
        query_whitened = self.transform(query_vector, year=year, embedding_type=embedding_type)
        results_whitened = self.transform(result_embeddings, year=year, embedding_type=embedding_type)

        # Compute whitened similarities
        similarities = np.dot(results_whitened, query_whitened)

        # Create mapping from location_id to similarity
        sim_map = dict(zip(embeddings_df['location_id'], similarities))

        # Update results with new similarities
        results = results.copy()
        results['similarity_original'] = results['similarity']
        results['similarity'] = results['location_id'].map(sim_map)

        # Remove rows where we couldn't compute whitened similarity
        results = results.dropna(subset=['similarity'])

        # Sort by new similarity
        results = results.sort_values('similarity', ascending=False)

        # Return top k if specified
        if top_k is not None:
            results = results.head(top_k)

        return results.reset_index(drop=True)

    def get_statistics_info(self, year: int, embedding_type: str = 'embedding') -> dict[str, Any] | None:
        """Get information about whitening statistics.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            Dictionary with statistics info or None if doesn't exist
        """
        stats_path = self._get_stats_path(year, embedding_type)

        if not stats_path.exists():
            return None

        stats = self.load_statistics(year, embedding_type)

        return {
            'year': stats.year,
            'n_samples': stats.n_samples,
            'n_components': stats.n_components,
            'created_at': stats.created_at,
            'explained_variance_ratio': (
                stats.explained_variance / stats.explained_variance.sum()
            ).tolist()[:10],  # First 10 components
            'file_path': str(stats_path),
            'file_size_mb': stats_path.stat().st_size / (1024 * 1024)
        }

    def list_statistics(self) -> list[dict[str, Any]]:
        """List all available whitening statistics.

        Returns:
            List of info dictionaries sorted by year
        """
        stats_list = []

        # Find all pickle files
        pattern = f"{self.universe_name}_*_whitening_*.pkl"
        for stats_file in self.stats_dir.glob(pattern):
            try:
                # Parse filename
                parts = stats_file.stem.split('_')
                if len(parts) >= 4:
                    year = int(parts[-1])
                    embedding_type = '_'.join(parts[1:-2])

                    info = self.get_statistics_info(year, embedding_type)
                    if info:
                        stats_list.append(info)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse stats file: {stats_file}: {e}")

        return sorted(stats_list, key=lambda x: x['year'])

    def delete_statistics(self, year: int, embedding_type: str = 'embedding') -> bool:
        """Delete whitening statistics.

        Args:
            year: Year
            embedding_type: Type of embedding

        Returns:
            True if deleted, False if didn't exist
        """
        stats_path = self._get_stats_path(year, embedding_type)

        if stats_path.exists():
            stats_path.unlink()
            logger.info(f"Deleted whitening statistics: {stats_path}")

            # Clear current stats if deleted
            if self._current_year == year:
                self._current_stats = None
                self._current_year = None

            return True

        return False
