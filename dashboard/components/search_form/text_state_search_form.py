"""State text search form for text-to-image state search (year-based)."""

from dash import Input, Output, State
import dash_bootstrap_components as dbc

from .base_search_form import BaseSearchForm
from ... import state

import logging
logger = logging.getLogger(__name__)


class TextStateSearchForm(BaseSearchForm):
    """Form for TextToImage state search.

    This form provides:
    - Text input
    - Year selector
    - Target year selector (optional)
    - Limit, media type, and search options
    - Search button
    """

    def __init__(self, available_years: list, all_streets: list = None):
        """Initialize the state text search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: Not used for text search, but kept for consistency
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            id_prefix='state-text-search-form',
            title='Text-to-Image State Search'
        )

    def _input_selector(self):
        """Return text input for text-based search."""
        return self._text_input()

    def _query_inputs(self) -> list:
        """Year and optional target year inputs."""
        return [
            # Year selector
            dbc.Col([
                *self._year_selector('year-selector', 'Year')
            ], width=2),

            # Target year (optional)
            dbc.Col([
                *self._year_selector('target-year-selector', 'Target Year (optional)')
            ], width=2),
        ]

    def execute_search(self, state, text, year, target_year, limit, media_type, **kwargs):
        """Execute state text search (text-to-image by year).

        Args:
            state: Application state module
            text: Search text query
            year: Year of the query
            target_year: Optional target year filter
            limit: Maximum number of results
            media_type: Type of media to search

        Returns:
            StateResultsSet with enriched results
        """
        from streettransformer.db.database import get_connection
        from streettransformer.query.queries.ask import TextToImageStateQuery

        # Configuration settings
        use_faiss_enabled = True
        use_whitening_enabled = False

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = TextToImageStateQuery(
            config=state.CONFIG,
            db=state.DB,
            text=text,
            year=year,
            target_years=[target_year] if target_year else None,
            limit=limit,
            media_types=[selected_media_type],
            use_faiss=use_faiss_enabled,
            use_whitening=use_whitening_enabled
        )

        results_set = query.search()

        # Enrich results with street names and image paths
        if len(results_set) > 0:
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                for result in results_set:
                    result.enrich_street_names(con, state.CONFIG.universe_name)
                    result.enrich_image_path(con, state.CONFIG.universe_name, selected_media_type)

        return results_set

    def register_callbacks(self, app):
        """Register callbacks for the state text search form.

        For text search, no street filtering is needed.
        """
        # No callbacks needed for text search form
        # Text input is handled directly in the search callback
        pass
