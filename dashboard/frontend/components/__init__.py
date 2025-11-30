"""Reusable UI components for the dashboard."""

from .results import format_results_accordion, format_change_results_accordion
from .details import render_location_details
#from .search_bar import create_search_fields

__all__ = [
    'format_results_accordion',
    'format_change_results_accordion',
    'render_location_details',
    'create_search_fields',
]
