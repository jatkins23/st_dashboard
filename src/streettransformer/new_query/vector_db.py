"""A class to handle translating the npy/npz files into a database when necessary"""

from pathlib import Path
from typing import Optional

from dataclasses import dataclass
import pandas as pd

import logging

LOCATION_ID = 'loc_key'
CHANGE_NAME = 'delta'
MASK_NAME = 'mask'
SBS_NAME = 'fusion'

@dataclass(frozen=True)
class SearchHit:
    image_path: str
    location_key: str
    year: int
    similarity: float
    distance: float
    mask_path: Optional[str] = None
    mask_stats: Optional[dict[str, float]] = None

class VectorDB:
    logger = logging.getLogger(__name__)
    
    def __init__(self, root_path: Path, meta_pq_path: Path = 'meta.parquet', location_id=LOCATION_ID, change_name=CHANGE_NAME, mask_name=MASK_NAME, sbs_name=SBS_NAME):
        self.root_path = root_path
        self.meta_df = pd.read_parquet(meta_pq_path)
        self.location_id = location_id
        self.change_name = change_name
        self.mask_name = mask_name
        self.sbs_name = sbs_name
        
    def query_by_location(
        location_key: str,
        year: int,
        db_cfg: dict[str, object], # change
        *,
        top_k: int,
        dissimilar: bool, # remove: split these into two functions
        vector_dim: int, # hardcode?
        column: str # ?
    ) -> list[SearchHit]:
          with VectorDB(vector_dimension=vector_dim, **db_cfg) as db:
              rows = db.fetch_embeddings_by_location
        
    
    