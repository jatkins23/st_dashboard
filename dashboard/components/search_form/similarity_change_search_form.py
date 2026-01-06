"""Change search form for image-to-image change detection."""

from dash import Input, Output, State
import dash_mantine_components as dmc

from .base_search_form import BaseSearchForm
from .utils import filter_street_options_by_selection, get_location_from_streets
from streettransformer.query.queries import ChangeSimilarityQuery
from streettransformer.db.database import get_connection

import logging
logger = logging.getLogger(__name__)


class ImageChangeSearchForm(BaseSearchForm):
    """Form for ImageToImage change search.

    This form provides:
    - Street selector
    - From year selector
    - To year selector
    - Sequential checkbox
    - Limit, media type, and search options
    - Search button
    """

    SEARCH_TYPE = 'change-similarity'
    TAB_LABEL = 'Change Similarity'
    QUERY_CLASS = ChangeSimilarityQuery
    RESULT_TYPE = 'change'

    def __init__(self, available_years: list, all_streets: list = None, all_boroughs: list = None):
        """Initialize the change search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            all_boroughs: List of all unique borough names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs,
            id_prefix='change-search-form',
            title='Image-to-Image Change Search'
        )

    def _input_selector(self):
        """Return street selector for image-based search."""
        return self._street_selector()

    def _query_inputs(self) -> list:
        """Year-from, year-to, and sequential checkbox inputs."""
        return [
            # From year
            dmc.GridCol(
                self._year_selector('year-from-selector', 'From Year'),
                span=2
            ),

            # To year
            dmc.GridCol(
                self._year_selector('year-to-selector', 'To Year'),
                span=2
            ),

            # Sequential checkbox
            dmc.GridCol(
                dmc.Switch(
                    id=self.Id('sequential-checkbox'),
                    label='Sequential',
                    checked=False,
                    size="sm"
                ),
                span=1
            ),
        ]

    def execute_search(self, app_ctx, location_id, year_from, year_to, limit, media_type, sequential, use_faiss: bool, use_whitening: bool, boroughs=None, **kwargs):
        """Execute change search (temporal change detection).

        Args:
            app_ctx: Application context module
            location_id: Location to search from
            year_from: Starting year for change detection
            year_to: Ending year for change detection
            limit: Maximum number of results
            media_type: Type of media to search
            sequential: Whether to require sequential years
            boroughs: Optional list of boroughs to filter by

        Returns:
            ChangeResultsSet with enriched results
        """

        # Default to 'image' if no media type selected
        selected_media_type = media_type if media_type else 'image'

        # Create and execute query
        query = ChangeSimilarityQuery(
            config=app_ctx.CONFIG,
            db=app_ctx.DB,
            location_id=location_id,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
            media_types=[selected_media_type],
            sequential=sequential,
            use_faiss=use_faiss,
            use_whitening=use_whitening,
            remove_self=True
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
        """Register callbacks for the change search form.

        Registers:
        1. Street filtering callback - updates available street options
        2. Location selection callback - converts streets to location_id
        """
        # Store the full street list for resetting
        all_streets_data = [{"label": s, "value": s} for s in self.all_streets]

        @app.callback(
            Output(self.Id('street-selector'), 'data'),
            Input(self.Id('street-selector'), 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def filter_street_options_change(selected_streets):
            """Filter street options to only show valid combinations."""
            from ... import context as app_ctx
            logger.info(f"Change street filter callback triggered. Selected: {selected_streets}")
            # Always pass the full street list, not the current filtered data
            result = filter_street_options_by_selection(selected_streets, all_streets_data, app_ctx)
            logger.info(f"Filtered options count: {len(result) if result else 0}")
            return result

        @app.callback(
            Output('selected-location-id', 'data', allow_duplicate=True),
            Input(self.Id('street-selector'), 'value'),
            prevent_initial_call='initial_duplicate'
        )
        def update_selected_location_change(selected_streets):
            """Convert selected streets to location_id."""
            from ... import context as app_ctx
            # Only update if at least 2 streets are selected
            if not selected_streets or len(selected_streets) < 2:
                return None
            return get_location_from_streets(selected_streets, app_ctx)
