"""State text search form for text-to-image state search (year-based)."""

from dash import Input, Output, State
import dash_mantine_components as dmc

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

    def __init__(self, available_years: list, all_streets: list = None, all_boroughs: list = None):
        """Initialize the state text search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: Not used for text search, but kept for consistency
            all_boroughs: List of all unique borough names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs,
            id_prefix='state-text-search-form',
            title='Text-to-Image State Search'
        )

    def _input_selector(self):
        """Return text input for text-based search."""
        return self._text_input()

    def _query_inputs(self) -> list:
        """Target year input only (no base year for text search)."""
        return [
            # Target year (optional)
            dmc.GridCol(
                self._year_selector('target-year-selector', 'Target Year (optional)'),
                span=2
            ),
        ]

    def execute_search(self, state, text, target_year, limit, media_type, boroughs=None, **kwargs):
        """Execute state text search (text-to-image by year).

        Args:
            state: Application state module
            text: Search text query
            target_year: Optional target year filter
            limit: Maximum number of results
            media_type: Type of media to search
            boroughs: Optional list of boroughs to filter by

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

        # Filter by borough if specified
        if boroughs and len(boroughs) > 0:
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                # Get borough for each result location
                location_ids = [r.location_id for r in results_set]
                if location_ids:
                    query = f"""
                        SELECT location_id, boro
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE location_id IN ({','.join(map(str, location_ids))})
                    """
                    boro_df = con.execute(query).df()
                    boro_map = dict(zip(boro_df['location_id'], boro_df['boro']))

                    # Filter results to only those in selected boroughs
                    filtered_results = [r for r in results_set if boro_map.get(r.location_id) in boroughs]
                    results_set.results = filtered_results

        return results_set

    def register_callbacks(self, app):
        """Register callbacks for the state text search form.

        For text search, no street filtering is needed.
        """
        # No callbacks needed for text search form
        # Text input is handled directly in the search callback
        pass
