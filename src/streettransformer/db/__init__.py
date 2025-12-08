"""Database and storage subpackage for streettransformer."""

from .database import Database, get_connection
from .embedding_db import EmbeddingDB, MediaEmbedding, ImageEmbedding
from .npz_cache import NPZCache, CacheInfo
from .faiss_index import FAISSIndexer, IndexInfo
from .whitening import WhiteningTransform, WhiteningStats

__all__ = [
    "Database",
    "get_connection",
    "EmbeddingDB",
    "MediaEmbedding",
    "ImageEmbedding",  # Backward compatibility alias
    "NPZCache",
    "CacheInfo",
    "FAISSIndexer",
    "IndexInfo",
    "WhiteningTransform",
    "WhiteningStats",
]
