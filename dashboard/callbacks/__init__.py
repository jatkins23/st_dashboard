"""Dash callbacks for the dashboard."""

from .ui import register_ui_callbacks
from .search import register_search_callbacks
from .details import register_details_callbacks
from .map import register_map_callbacks


def register_all_callbacks(app):
    """Register all dashboard callbacks.

    Args:
        app: Dash app instance
    """
    register_ui_callbacks(app)
    register_search_callbacks(app)
    register_details_callbacks(app)
    register_map_callbacks(app)
