"""StreetTransformer: Vector embeddings for street imagery analysis.

A standalone package for managing, indexing, and searching CLIP embeddings
of street-level imagery stored in DuckDB.

Example:
    >>> from streettransformer import Config, EmbeddingDB
    >>>
    >>> # Initialize with existing database
    >>> config = Config(
    ...     database_path="/path/to/data.db",
    ...     universe_name="lion"
    ... )
    >>>
    >>> # Setup schema (first time only)
    >>> db = EmbeddingDB(config)
    >>> db.setup_schema()
    >>>
    >>> # Search for similar images
    >>> import numpy as np
    >>> query = np.random.rand(512)
    >>> results = db.search_similar(query, limit=10, year=2020)
"""

__version__ = "0.1.0"

from .config import Config
from .database import Database, get_connection
from .embedding_db import EmbeddingDB, ImageEmbedding
from .npz_cache import NPZCache, CacheInfo
from .faiss_index import FAISSIndexer, IndexInfo
from .whitening import WhiteningTransform, WhiteningStats
from .clip_encoding import CLIPEncoder

# Query classes (new)
from .query import (
    BaseQuery,
    StateLocationQuery,
    ChangeLocationQuery,
    StateTextQuery,
)

__all__ = [
    # Config and database
    "Config",
    "Database",
    "get_connection",

    # Core embedding storage
    "EmbeddingDB",
    "ImageEmbedding",

    # Caching
    "NPZCache",
    "CacheInfo",

    # FAISS indexing
    "FAISSIndexer",
    "IndexInfo",

    # Whitening
    "WhiteningTransform",
    "WhiteningStats",

    # CLIP encoding
    "CLIPEncoder",

    # Query classes
    "BaseQuery",
    "StateLocationQuery",
    "ChangeLocationQuery",
    "StateTextQuery",
]
