"""Change text search form for text-to-image change detection."""

from dash import Input, Output, State
import dash_mantine_components as dmc

from .base_search_form import BaseSearchForm
from streettransformer.query.queries import ChangeDescriptionQuery
from streettransformer.db.database import get_connection

import logging
logger = logging.getLogger(__name__)


class ChangeDescriptionSearchForm(BaseSearchForm):
    """Form for TextToImage change search.

    This form provides:
    - Text input
    - From year selector
    - To year selector
    - Sequential checkbox
    - Limit, media type, and search options
    - Search button
    """

    SEARCH_TYPE = 'change-description'
    TAB_LABEL = 'Change Description'
    QUERY_CLASS = ChangeDescriptionQuery
    RESULT_TYPE = 'change'

    def __init__(self, available_years: list, all_streets: list = None, all_boroughs: list = None):
        """Initialize the change text search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: Not used for text search, but kept for consistency
            all_boroughs: List of all unique borough names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs,
            id_prefix='change-text-search-form',
            title='Text-to-Image Change Search'
        )

    def _input_selector(self):
        """Return text input for text-based search."""
        return self._text_input()

    def _query_inputs(self) -> list:
        """No year inputs for text change search."""
        return []

    def execute_search(self, app_ctx, text, limit, media_type, sequential, boroughs=None, **kwargs):
        """Execute change text search (text-to-image temporal change detection).

        Args:
            app_ctx: Application app_ctx module
            text: Search text query
            limit: Maximum number of results
            media_type: Type of media to search
            sequential: Whether to require sequential years
            boroughs: Optional list of boroughs to filter by

        Returns:
            ChangeResultsSet with enriched results
        """
        # Configuration settings
        use_faiss_enabled = True
        use_whitening_enabled = False

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = ChangeDescriptionQuery(
            config=app_ctx.CONFIG,
            db=app_ctx.DB,
            text=text,
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
            with get_connection(app_ctx.CONFIG.database_path, read_only=True) as con:
                for result in results_set:
                    result.enrich_street_names(con, app_ctx.CONFIG.universe_name)

        # Filter by borough if specified
        if boroughs and len(boroughs) > 0:
            with get_connection(app_ctx.CONFIG.database_path, read_only=True) as con:
                # Get borough for each result location
                location_ids = [r.location_id for r in results_set]
                if location_ids:
                    query = f"""
                        SELECT location_id, boro
                        FROM {app_ctx.CONFIG.universe_name}.locations
                        WHERE location_id IN ({','.join(map(str, location_ids))})
                    """
                    boro_df = con.execute(query).df()
                    boro_map = dict(zip(boro_df['location_id'], boro_df['boro']))

                    # Filter results to only those in selected boroughs
                    filtered_results = [r for r in results_set if boro_map.get(r.location_id) in boroughs]
                    results_set.results = filtered_results

        return results_set

    def register_callbacks(self, app):
        """Register callbacks for the change text search form.

        For text search, no street filtering is needed.
        """
        # No callbacks needed for text search form
        # Text input is handled directly in the search callback
        pass
