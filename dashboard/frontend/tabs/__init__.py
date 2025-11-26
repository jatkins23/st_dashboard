"""Dashboard tab components."""

from .location import create_location_search_tab
from .text import create_text_search_tab
from .change import create_change_search_tab
from .stats import create_stats_tab

__all__ = [
    'create_location_search_tab',
    'create_text_search_tab',
    'create_change_search_tab',
    'create_stats_tab',
]
