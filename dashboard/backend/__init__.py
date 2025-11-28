"""Backend logic for dashboard searches and data operations."""

from .search import (
    search_by_location,
    search_by_text,
    search_change_patterns,
    get_embedding_stats
)

from .results import (
    BaseQueryResult,
    DisplayMixin,
    StateLocationResult,
    ChangeLocationResult,
)

__all__ = [
    # Search functions
    'search_by_location',
    'search_by_text',
    'search_change_patterns',
    'get_embedding_stats',

    # Result classes
    'BaseQueryResult',
    'DisplayMixin',
    'StateLocationResult',
    'ChangeLocationResult',
]
