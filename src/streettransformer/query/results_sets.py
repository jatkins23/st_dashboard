
#### ---- QueryResultsSet ---- ####

import pandas as pd
from typing import Optional

from pydantic import BaseModel
from .results_instances import QueryResultInstance
from .metadata import QueryMetadata

class QueryResultsSet(BaseModel):
    """Shared Functionality for ResultsSets"""
    results: list[QueryResultInstance]
    metadata: QueryMetadata

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def df(self) -> pd.DataFrame:
        """Convert to DataFrame"""
        return pd.DataFrame([r.model_dump() for r in self.results])
    
    def __getitem__(self, idx: int) -> QueryResultInstance:
        return iter(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def __len__(self) -> int:
        return len(self.results)
    
class StateResultsSet(QueryResultsSet):
    query_location_id: str
    query_year: int
    target_years: Optional[list[int]]=None

class ChangeResultsSet(QueryResultsSet):
    query_location_id: str
    query_year_from: int
    query_year_to: int
    target_years: Optional[list[int]]=None
    sequential: bool=False
