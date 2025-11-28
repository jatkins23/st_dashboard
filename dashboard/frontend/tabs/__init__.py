"""Dashboard tab components."""

from .details import create_details_tab
from .state import create_state_search_tab as create_location_search_tab
from .text import create_text_search_tab
from .change import create_change_search_tab
from .stats import create_stats_tab

__all__ = [
    'create_details_tab',
    'create_location_search_tab',
    'create_text_search_tab',
    'create_change_search_tab',
    'create_stats_tab',
]
