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

from .config import STConfig
from .db.database import Database, get_connection
from .db.embedding_db import EmbeddingDB, MediaEmbedding, ImageEmbedding
from .db.npz_cache import NPZCache, CacheInfo
from .db.faiss_index import FAISSIndexer, IndexInfo
from .db.whitening import WhiteningTransform, WhiteningStats
from .query import (
    CLIPEncoder,
    ChangeSimilarityQuery,
    StateSimilarityQuery,
    StateResultsSet,
    ChangeResultsSet,
    StateResultInstance,
    ChangeResultInstance,
    StateMixin,
    ChangeMixin,
    DatabaseMixin,
    SearchMethodMixin
)

__all__ = [
    # Config and database
    "STConfig",
    "Database",
    "get_connection",

    # Core embedding storage
    "EmbeddingDB",
    "MediaEmbedding",
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
    
    # query classes
    'ChangeSimilarityQuery',
    'StateSimilarityQuery',
    
    # ResultsSet Classes
    'StateResultsSet',
    'ChangeResultsSet',

    # ResultInstance Classes
    'StateResultInstance',
    'ChangeResultInstance',
    
    # Mixins (for advanced usage)
    'StateMixin',
    'ChangeMixin',
    'DatabaseMixin',
    'SearchMethodMixin',
]
