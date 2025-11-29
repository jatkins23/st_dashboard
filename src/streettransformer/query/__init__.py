"""Query classes for streettransformer.

This package provides a clean abstraction for executing different types of searches:
- StateLocationQuery: Find similar locations at a point in time
- ChangeLocationQuery: Find locations with similar change patterns
- StateTextQuery: Find locations matching text descriptions

Example:
    >>> from streettransformer import STConfig, EmbeddingDB
    >>> from streettransformer.query import StateLocationQuery
    >>>
    >>> config = STConfig(database_path="data.db", universe_name="lion")
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

from .queries import ImageToImageStateQuery, ImageToImageChangeQuery, StateResultInstance, ChangeResultInstance, StateResultsSet, ChangeResultsSet
from .mixins import StateMixin, ChangeMixin, DatabaseMixin, SearchMethodMixin
from .clip_embedding import CLIPEncoder

__all__ = [
    'CLIPEncoder',

    # Concrete query classes
    'ImageToImageStateQuery',
    'ImageToImageChangeQuery',
    
    # Concrete ResultsSet Classes
    'StateResultsSet',
    'ChangeResultsSet',

    # Concrete ResultInstance Classes
    'StateResultInstance',
    'ChangeResultInstance',
    
    # Mixins (for advanced usage)
    'StateMixin',
    'ChangeMixin',
    'DatabaseMixin',
    'SearchMethodMixin'
]
