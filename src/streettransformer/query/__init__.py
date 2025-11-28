"""Query classes for streettransformer.

This package provides a clean abstraction for executing different types of searches:
- StateLocationQuery: Find similar locations at a point in time
- ChangeLocationQuery: Find locations with similar change patterns
- StateTextQuery: Find locations matching text descriptions

Example:
    >>> from streettransformer import Config, EmbeddingDB
    >>> from streettransformer.query import StateLocationQuery
    >>>
    >>> config = Config(database_path="data.db", universe_name="lion")
    >>> db = EmbeddingDB(config)
    >>>
    >>> # Search for similar locations
    >>> query = StateLocationQuery(
    ...     location_id=123,
    ...     year=2020,
    ...     config=config,
    ...     db=db,
    ...     limit=10,
    ...     use_faiss=True
    ... )
    >>> results = query.execute()
"""

from .base import BaseQuery
from .queries import StateLocationQuery, ChangeLocationQuery, StateTextQuery
from .mixins import StateMixin, ChangeMixin, DatabaseMixin, SearchMethodMixin

__all__ = [
    # Base classes
    'BaseQuery',

    # Concrete query classes
    'StateLocationQuery',
    'ChangeLocationQuery',
    'StateTextQuery',

    # Mixins (for advanced usage)
    'StateMixin',
    'ChangeMixin',
    'DatabaseMixin',
    'SearchMethodMixin',
]
