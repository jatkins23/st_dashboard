"""Change text search form for text-to-image change detection."""

from dash import Input, Output, State
import dash_mantine_components as dmc

from .base_search_form import BaseSearchForm
from ... import state

import logging
logger = logging.getLogger(__name__)


class TextChangeSearchForm(BaseSearchForm):
    """Form for TextToImage change search.

    This form provides:
    - Text input
    - From year selector
    - To year selector
    - Sequential checkbox
    - Limit, media type, and search options
    - Search button
    """

    def __init__(self, available_years: list, all_streets: list = None):
        """Initialize the change text search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: Not used for text search, but kept for consistency
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            id_prefix='change-text-search-form',
            title='Text-to-Image Change Search'
        )

    def _input_selector(self):
        """Return text input for text-based search."""
        return self._text_input()

    def _query_inputs(self) -> list:
        """No year inputs for text change search."""
        return []

    def execute_search(self, state, text, year_from, year_to, limit, media_type, sequential, **kwargs):
        """Execute change text search (text-to-image temporal change detection).

        Args:
            state: Application state module
            text: Search text query
            year_from: Starting year for change detection
            year_to: Ending year for change detection
            limit: Maximum number of results
            media_type: Type of media to search
            sequential: Whether to require sequential years

        Returns:
            ChangeResultsSet with enriched results
        """
        from streettransformer.db.database import get_connection
        from streettransformer.query.queries.ask import TextToImageChangeQuery

        # Configuration settings
        use_faiss_enabled = True
        use_whitening_enabled = False

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = TextToImageChangeQuery(
            config=state.CONFIG,
            db=state.DB,
            text=text,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
            media_types=[selected_media_type],
            sequential=sequential,
            use_faiss=use_faiss_enabled,
            use_whitening=use_whitening_enabled
        )

        results_set = query.search()

        # Enrich results with street names
        # Note: Change results have before_path and after_path already set by the query
        if len(results_set) > 0:
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                for result in results_set:
                    result.enrich_street_names(con, state.CONFIG.universe_name)

        return results_set

    def register_callbacks(self, app):
        """Register callbacks for the change text search form.

        For text search, no street filtering is needed.
        """
        # No callbacks needed for text search form
        # Text input is handled directly in the search callback
        pass
