"""Change search form for image-to-image change detection."""

from dash import Input, Output, State
import dash_bootstrap_components as dbc

from .base_search_form import BaseSearchForm
from .utils import filter_street_options_by_selection
from ... import state

import logging
logger = logging.getLogger(__name__)


class ChangeSearchForm(BaseSearchForm):
    """Form for ImageToImage change search.

    This form provides:
    - Street selector
    - From year selector
    - To year selector
    - Sequential checkbox
    - Limit, media type, and search options
    - Search button
    """

    def __init__(self, available_years: list, all_streets: list = None):
        """Initialize the change search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            id_prefix='change-search-form',
            title='Image-to-Image Change Search'
        )

    def _query_inputs(self) -> list:
        """Year-from, year-to, and sequential checkbox inputs."""
        return [
            # From year
            dbc.Col([
                *self._year_selector('year-from-selector', 'From Year')
            ], width=2),

            # To year
            dbc.Col([
                *self._year_selector('year-to-selector', 'To Year')
            ], width=2),

            # Sequential checkbox
            dbc.Col([
                dbc.Label("Sequential", size='sm'),
                dbc.Checklist(
                    id=self.Id('sequential-checkbox'),
                    options=[{'label': ' Sequential', 'value': 'sequential'}],
                    value=[],
                    switch=True
                )
            ], width=1),
        ]

    def execute_search(self, state, location_id, year_from, year_to, limit, media_type, sequential, **kwargs):
        """Execute change search (temporal change detection).

        Args:
            state: Application state module
            location_id: Location to search from
            year_from: Starting year for change detection
            year_to: Ending year for change detection
            limit: Maximum number of results
            media_type: Type of media to search
            sequential: Whether to require sequential years

        Returns:
            ChangeResultsSet with enriched results
        """
        from streettransformer.db.database import get_connection
        from streettransformer.query.queries.ask import ImageToImageChangeQuery

        # Configuration settings
        use_faiss_enabled = True
        use_whitening_enabled = False

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = ImageToImageChangeQuery(
            config=state.CONFIG,
            db=state.DB,
            location_id=location_id,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
            media_types=[selected_media_type],
            sequential=sequential,
            use_faiss=use_faiss_enabled,
            use_whitening=use_whitening_enabled,
            remove_self=True
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
        """Register callbacks for the change search form.

        Registers:
        1. Street filtering callback - updates available street options
        2. Location selection callback - converts streets to location_id
        """
        from dash import Input, Output, State
        from .utils import filter_street_options_by_selection, get_location_from_streets
        from ... import state

        @app.callback(
            Output(self.Id('street-selector'), 'data'),
            Input(self.Id('street-selector'), 'value'),
            State(self.Id('street-selector'), 'data'),
            prevent_initial_call='initial_duplicate'
        )
        def filter_street_options_change(selected_streets, current_data):
            """Filter street options to only show valid combinations."""
            logger.info(f"Change street filter callback triggered. Selected: {selected_streets}")
            result = filter_street_options_by_selection(selected_streets, current_data, state)
            logger.info(f"Filtered options count: {len(result) if result else 0}")
            return result

        @app.callback(
            Output('selected-location-id', 'data', allow_duplicate=True),
            Input(self.Id('street-selector'), 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_selected_location_change(selected_streets):
            """Convert selected streets to location_id."""
            # Only update if at least 2 streets are selected
            if not selected_streets or len(selected_streets) < 2:
                return None
            return get_location_from_streets(selected_streets, state)
