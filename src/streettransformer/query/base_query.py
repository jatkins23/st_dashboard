from typing import Any
from pydantic import BaseModel, ConfigDict

from .mixins import SearchMethodMixin, DatabaseMixin
from .results_sets import QueryResultsSet


class BaseQuery(DatabaseMixin, SearchMethodMixin, BaseModel):
    # use_faiss: bool = True
    # use_whitening: bool = False
    distance_metric: str = 'cosine'

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search(self, **kwargs: Any) -> QueryResultsSet:
        ...

    def get_cache_key(self) -> str:

        """Get unique cache key for this query.

        Returns:
            String like: loc_123_change2015to2020_n10
        """
        ...
