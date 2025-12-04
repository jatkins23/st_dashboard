"""FAISS-based indexing for fast vector similarity search.

This module provides FAISS (Facebook AI Similarity Search) indexing for embeddings,
offering much faster search than DuckDB for large datasets.

Supported index types:
- Flat: Exact search, slowest but most accurate
- IVFFlat: Inverted file index with flat quantization
- IVFPQ: Inverted file index with product quantization (compressed)
- HNSW: Hierarchical Navigable Small World graph (fast approximate search)

Usage:
    >>> config = STConfig(database_path="data.db", universe_name="lion")
    >>> indexer = FAISSIndexer(config)
    >>> indexer.build_index(year=2020, index_type='hnsw')
    >>>
    >>> # Search using FAISS
    >>> results = indexer.search(
    ...     query_vector=query_embedding,
    ...     k=10,
    ...     year=2020
    ... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import logging
from dataclasses import dataclass
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:
    faiss = None

from .npz_cache import NPZCache
from ..config import STConfig
from .database import get_connection

logger = logging.getLogger(__name__)

IndexType = Literal['flat', 'ivf_flat', 'ivf_pq', 'hnsw']


@dataclass
class IndexInfo:
    """Information about a FAISS index.

    Attributes:
        year: Year of indexed embeddings
        index_type: Type of FAISS index
        num_vectors: Number of vectors in index
        vector_dim: Dimensionality of vectors
        file_size_mb: Size of index file in megabytes
        created_at: ISO timestamp of index creation
        index_path: Path to index file
        metadata_path: Path to metadata file
    """
    year: int
    index_type: str
    num_vectors: int
    vector_dim: int
    file_size_mb: float
    created_at: str
    index_path: Path
    metadata_path: Path


class FAISSIndexer:
    """Manage FAISS indexes for embeddings.

    Attributes:
        config: Configuration with database path and settings
        universe_name: Name of the universe
        index_dir: Directory for index files
        vector_dim: Embedding dimensionality
        cache: NPZ cache for loading embeddings
    """

    def __init__(self, config: STConfig):
        """Initialize FAISS indexer.

        Args:
            config: Configuration with database path and settings

        Raises:
            ImportError: If faiss is not installed
        """
        if faiss is None:
            raise ImportError(
                "faiss is required for FAISS indexing. "
                "Install with: pip install faiss-cpu  (or faiss-gpu for GPU support)"
            )

        self.config = config
        self.universe_name = config.universe_name
        self.index_dir = config.index_dir
        self.vector_dim = config.vector_dim
        self.cache = NPZCache(config)

        # Currently loaded index
        self._current_index: faiss.Index | None = None
        self._current_metadata: pd.DataFrame | None = None
        self._current_year: int | None = None
        self._current_type: str | None = None

    def _get_index_path(self, year: int, index_type: str = 'hnsw') -> Path:
        """Get path to index file.

        Args:
            year: Year
            index_type: Type of index

        Returns:
            Path to index file
        """
        filename = f"{self.universe_name}_{index_type}_{year}.faiss"
        return self.index_dir / filename

    def _get_metadata_path(self, year: int, index_type: str = 'hnsw') -> Path:
        """Get path to metadata file.

        Args:
            year: Year
            index_type: Type of index

        Returns:
            Path to metadata pickle file
        """
        filename = f"{self.universe_name}_{index_type}_{year}_metadata.pkl"
        return self.index_dir / filename

    def _create_index(
        self,
        index_type: IndexType,
        n_vectors: int,
        nlist: int | None = None,
        m_pq: int = 8,
        nbits_pq: int = 8,
        m_hnsw: int = 32,
        ef_construction: int = 40
    ) -> faiss.Index:
        """Create a FAISS index of the specified type.

        Args:
            index_type: Type of index to create
            n_vectors: Number of vectors that will be indexed
            nlist: Number of clusters for IVF indexes (default: sqrt(n_vectors))
            m_pq: Number of subquantizers for PQ (must divide vector_dim)
            nbits_pq: Number of bits per subquantizer for PQ
            m_hnsw: Number of neighbors for HNSW graph
            ef_construction: Size of dynamic candidate list for HNSW

        Returns:
            FAISS index
        """
        if nlist is None:
            nlist = int(np.sqrt(n_vectors))
            nlist = max(nlist, 100)  # Minimum 100 clusters

        if index_type == 'flat':
            # Exact search using L2 distance
            index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity

        elif index_type == 'ivf_flat':
            # IVF with flat quantization (no compression)
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist, faiss.METRIC_INNER_PRODUCT)

        elif index_type == 'ivf_pq':
            # IVF with product quantization (compressed)
            quantizer = faiss.IndexFlatIP(self.vector_dim)
            index = faiss.IndexIVFPQ(
                quantizer,
                self.vector_dim,
                nlist,
                m_pq,
                nbits_pq,
                faiss.METRIC_INNER_PRODUCT
            )

        elif index_type == 'hnsw':
            # Hierarchical Navigable Small World graph
            index = faiss.IndexHNSWFlat(self.vector_dim, m_hnsw, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        logger.info(f"Created {index_type} index for {n_vectors} vectors (dim={self.vector_dim})")
        return index

    def build_index(
        self,
        year: int,
        index_type: IndexType = 'hnsw',
        embedding_type: str = 'embedding',
        force_rebuild: bool = False,
        use_cache: bool = True,
        **index_params: Any
    ) -> Path:
        """Build FAISS index from embeddings.

        Args:
            year: Year to index
            index_type: Type of index to build
            embedding_type: Type of embedding to index
            force_rebuild: Force rebuild even if index exists
            use_cache: Use NPZ cache if available
            **index_params: Additional parameters for index creation

        Returns:
            Path to created index file
        """
        index_path = self._get_index_path(year, index_type)
        metadata_path = self._get_metadata_path(year, index_type)

        # Check if index already exists
        if index_path.exists() and not force_rebuild:
            logger.info(f"Index already exists: {index_path}")
            return index_path

        logger.info(f"Building {index_type} index for year {year}...")

        # Load embeddings (try cache first if enabled)
        if use_cache:
            try:
                data = self.cache.load(year, embedding_type)
                embeddings = data['embeddings']
                metadata = data['metadata']
                logger.info(f"Loaded embeddings from cache")
            except FileNotFoundError:
                logger.info(f"Cache not found, building from database...")
                self.cache.build_from_db(year=year, embedding_type=embedding_type)
                data = self.cache.load(year, embedding_type)
                embeddings = data['embeddings']
                metadata = data['metadata']
        else:
            # Load directly from database
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
                    WHERE year = {year}
                        AND {embedding_type} IS NOT NULL
                    ORDER BY location_id, media_type
                """).df()

            embeddings = np.stack(df[embedding_type].values).astype(np.float32)
            metadata = df[['location_id', 'location_key', 'year', 'media_type', 'path']]

        # Ensure embeddings are normalized for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Create index
        n_vectors = len(embeddings)
        index = self._create_index(index_type, n_vectors, **index_params)

        # Train index if needed (IVF indexes need training)
        if hasattr(index, 'is_trained') and not index.is_trained:
            logger.info(f"Training {index_type} index on {n_vectors} vectors...")
            index.train(embeddings)

        # Add vectors to index
        logger.info(f"Adding {n_vectors} vectors to index...")
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved index to {index_path}")

        # Save metadata
        metadata_dict = {
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'universe_name': self.universe_name,
            'embedding_type': embedding_type,
            'index_type': index_type,
            'vector_dim': self.vector_dim,
            'num_vectors': n_vectors
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)

        logger.info(f"Saved metadata to {metadata_path}")

        file_size_mb = index_path.stat().st_size / (1024 * 1024)
        logger.info(f"Index built successfully ({file_size_mb:.2f} MB)")

        return index_path

    def load_index(self, year: int, index_type: str = 'hnsw') -> None:
        """Load a FAISS index into memory.

        Args:
            year: Year
            index_type: Type of index

        Raises:
            FileNotFoundError: If index doesn't exist
        """
        index_path = self._get_index_path(year, index_type)
        metadata_path = self._get_metadata_path(year, index_type)

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_path}\n"
                f"Run build_index() first to create the index."
            )

        # Load index
        logger.info(f"Loading index from {index_path}...")
        index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)

        self._current_index = index
        self._current_metadata = metadata_dict['metadata']
        self._current_year = year
        self._current_type = index_type

        logger.info(
            f"Loaded {index_type} index for year {year} "
            f"({metadata_dict['num_vectors']} vectors)"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        year: int | None = None,
        index_type: str = 'hnsw',
        nprobe: int = 10,
        ef_search: int = 16
    ) -> pd.DataFrame:
        """Search for similar vectors using FAISS.

        Args:
            query_vector: Query embedding (will be normalized)
            k: Number of results to return
            year: Year to search (loads index if not already loaded)
            index_type: Type of index to use
            nprobe: Number of clusters to search for IVF indexes
            ef_search: Size of dynamic candidate list for HNSW

        Returns:
            DataFrame with columns: location_id, location_key, year, media_type, path, similarity

        Raises:
            ValueError: If year is None and no index is loaded
        """
        # Load index if needed
        if self._current_index is None or self._current_year != year or self._current_type != index_type:
            if year is None:
                raise ValueError("Must specify year when no index is loaded")
            self.load_index(year, index_type)

        # Normalize query vector
        query_vector = query_vector.astype(np.float32)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        # Reshape to (1, dim) for FAISS
        query_vector = query_vector.reshape(1, -1)

        # Set search parameters for IVF indexes
        if hasattr(self._current_index, 'nprobe'):
            self._current_index.nprobe = nprobe

        # Set search parameters for HNSW
        if hasattr(self._current_index, 'hnsw'):
            self._current_index.hnsw.efSearch = ef_search

        # Search
        similarities, indices = self._current_index.search(query_vector, k)

        # Build results DataFrame
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx >= 0:  # FAISS returns -1 for empty results
                row = self._current_metadata.iloc[idx]
                result = {
                    'location_id'   : row['location_id'],
                    'location_key'  : row['location_key'],
                    'year'          : row['year'],
                    'media_type'    : row['media_type'],
                    'path'          : row['path'],
                    'similarity'    : float(sim)
                }
                results.append(result)

        return pd.DataFrame(results)

    def get_index_info(self, year: int, index_type: str = 'hnsw') -> IndexInfo | None:
        """Get information about an index.

        Args:
            year: Year
            index_type: Type of index

        Returns:
            IndexInfo object or None if index doesn't exist
        """
        index_path = self._get_index_path(year, index_type)
        metadata_path = self._get_metadata_path(year, index_type)

        if not index_path.exists():
            return None

        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)

        return IndexInfo(
            year=year,
            index_type=index_type,
            num_vectors=metadata_dict['num_vectors'],
            vector_dim=metadata_dict['vector_dim'],
            file_size_mb=index_path.stat().st_size / (1024 * 1024),
            created_at=metadata_dict['created_at'],
            index_path=index_path,
            metadata_path=metadata_path
        )

    def list_indexes(self) -> list[IndexInfo]:
        """List all available indexes.

        Returns:
            List of IndexInfo objects sorted by year and index type
        """
        indexes = []

        # Find all FAISS index files
        pattern = f"{self.universe_name}_*.faiss"
        for index_file in self.index_dir.glob(pattern):
            try:
                # Parse filename to extract year and type
                # Format: {universe}_{type}_{year}.faiss
                parts = index_file.stem.split('_')
                if len(parts) >= 3:
                    year = int(parts[-1])
                    index_type = '_'.join(parts[1:-1])

                    info = self.get_index_info(year, index_type)
                    if info:
                        indexes.append(info)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse index file: {index_file}: {e}")

        return sorted(indexes, key=lambda x: (x.year, x.index_type))

    def delete_index(self, year: int, index_type: str = 'hnsw') -> bool:
        """Delete an index and its metadata.

        Args:
            year: Year
            index_type: Type of index

        Returns:
            True if deleted, False if didn't exist
        """
        index_path = self._get_index_path(year, index_type)
        metadata_path = self._get_metadata_path(year, index_type)

        deleted = False

        if index_path.exists():
            index_path.unlink()
            logger.info(f"Deleted index: {index_path}")
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Deleted metadata: {metadata_path}")
            deleted = True

        # Clear current index if it was deleted
        if self._current_year == year and self._current_type == index_type:
            self._current_index = None
            self._current_metadata = None
            self._current_year = None
            self._current_type = None

        return deleted
