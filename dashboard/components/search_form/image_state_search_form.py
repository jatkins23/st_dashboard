"""State search form for image-to-image state search (year-based)."""

from dash import Input, Output, State
import dash_mantine_components as dmc

from .base_search_form import BaseSearchForm
from .utils import filter_street_options_by_selection
from ... import state

import logging
logger = logging.getLogger(__name__)


class ImageStateSearchForm(BaseSearchForm):
    """Form for ImageToImage state search.

    This form provides:
    - Street selector
    - Year selector
    - Target year selector (optional)
    - Limit, media type, and search options
    - Search button
    """

    def __init__(self, available_years: list, all_streets: list = None, all_boroughs: list = None):
        """Initialize the state search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            all_boroughs: List of all unique borough names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs,
            id_prefix='state-search-form',
            title='Image-to-Image State Search'
        )

    def _input_selector(self):
        """Return street selector for image-based search."""
        return self._street_selector()

    def _query_inputs(self) -> list:
        """Year and optional target year inputs."""
        return [
            # Year selector
            dmc.GridCol(
                self._year_selector('year-selector', 'Year'),
                span=2
            ),

            # Target year (optional)
            dmc.GridCol(
                self._year_selector('target-year-selector', 'Target Year (optional)'),
                span=2
            ),
        ]

    def execute_search(self, state, location_id, year, target_year, limit, media_type, use_faiss:bool, use_whitening:bool, boroughs=None, **kwargs):
        """Execute state search (image-to-image by year).

        Args:
            state: Application state module
            location_id: Location to search from
            year: Year of the query image
            target_year: Optional target year filter
            limit: Maximum number of results
            media_type: Type of media to search
            boroughs: Optional list of boroughs to filter by

        Returns:
            StateResultsSet with enriched results
        """
        from streettransformer.db.database import get_connection
        from streettransformer.query.queries.ask import ImageToImageStateQuery

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = ImageToImageStateQuery(
            config=state.CONFIG,
            db=state.DB,
            location_id=location_id,
            year=year,
            target_years=[target_year] if target_year else None,
            limit=limit,
            media_types=[selected_media_type],
            use_faiss=use_faiss,
            use_whitening=use_whitening,
            remove_self=True
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
        """Register callbacks for the state search form.

        Registers:
        1. Street filtering callback - updates available street options
        2. Location selection callback - converts streets to location_id
        """
        from .utils import get_location_from_streets

        # Store the full street list for resetting
        all_streets_data = [{"label": s, "value": s} for s in self.all_streets]

        @app.callback(
            Output(self.Id('street-selector'), 'data'),
            Input(self.Id('street-selector'), 'value'),
            prevent_initial_call=False
        )
        def filter_street_options_state(selected_streets):
            """Filter street options to only show valid combinations."""
            logger.info(f"State street filter callback triggered. Selected: {selected_streets}")
            # Always pass the full street list, not the current filtered data
            result = filter_street_options_by_selection(selected_streets, all_streets_data, state)
            logger.info(f"Filtered options count: {len(result) if result else 0}")
            return result

        @app.callback(
            Output('selected-location-id', 'data'),
            Input(self.Id('street-selector'), 'value'),
            prevent_initial_call=False
        )
        def update_selected_location_state(selected_streets):
            """Convert selected streets to location_id."""
            # Only update if at least 2 streets are selected
            if not selected_streets or len(selected_streets) < 2:
                return None
            return get_location_from_streets(selected_streets, state)
