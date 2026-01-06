import argparse
from pathlib import Path
from typing import Iterable, Optional, Union, cast
import logging

import numpy as np

from ..image_retrieval.blip_embeddings import BLIPEmbedder
from ..image_retrieval.clip_embeddings import CLIPEmbedder
from ..image_retrieval.vector_db import ChangeVector, SearchHit, StoredEmbedding, VectorDB

EmbedderType = Union(CLIPEmbedder, BLIPEmbedder)

# make_embedder
def _make_embedder():
    pass

def _feature_to_column(name: str):
    pass

def _print_hits(
    title: str, hits: Iterable[SearchHit]
) -> None:
    pass


def _print_changes(
    title: str,
    rows: Iterable[ChangeVector],
    limit: int | None = None
) -> None:
    # move to Change Result Set
    pass

def query_by_location(self, vector_dim: int, db_cfg: dict[str, object]):
    """Execute Search from saved npz files"""
    logger = logging.getLogger(__name__)
    
    media_types = getattr(self, 'media_types', None) or ['image']
    if isinstance(media_types, str):
        media_types = [media_types]
        
        media_type_filter = "', '".join(media_types)
        
        with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
            metadata = db.fetch_metadata_for_path
        
    