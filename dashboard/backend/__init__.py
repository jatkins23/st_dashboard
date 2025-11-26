"""Backend logic for dashboard searches and data operations."""

from .search import (
    search_by_location,
    search_by_text,
    search_change_patterns,
    get_embedding_stats
)

__all__ = [
    'search_by_location',
    'search_by_text',
    'search_change_patterns',
    'get_embedding_stats',
]
