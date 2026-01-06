"""Map component for displaying locations and projects."""

from dash import dcc, Input, Output
from dash.development.base_component import Component as DashComponent

from .base import BaseComponent
from ..utils.map_utils import create_location_map

import logging
logger = logging.getLogger(__name__)

class Map(BaseComponent):
    """Map component that displays locations and projects.

    This component provides an interactive map showing:
    - All project locations
    - Selected query location (highlighted)
    - Search result locations (highlighted)
    """

    def __init__(
        self,
        id_prefix: str = 'map',
        center_lat: float = 40.7128,
        center_lon: float = -74.0060,
        zoom: int = 11,
        height: str = '60vh'
    ):
        """Initialize the map component.

        Args:
            id_prefix: Prefix for component IDs
            center_lat: Default center latitude
            center_lon: Default center longitude
            zoom: Default zoom level
            height: CSS height of the map
        """
        super().__init__(id_prefix=id_prefix)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.height = height

    def register_callbacks(self, app):
        """Register map update callback.

        Updates the map display when:
        - A query location is selected (shared across tabs)
        - Search results are returned
        - Active tab changes

        The map shows the shared selected location and tab-specific results.
        """

        @app.callback(
            Output('main-map', 'figure'),
            Input('active-search-tab', 'data'),
            Input('selected-location-id', 'data'),
            Input('state-similarity-result-locations', 'data'),
            Input('state-description-result-locations', 'data'),
            Input('change-similarity-result-locations', 'data'),
            Input('change-description-result-locations', 'data'),
            Input('dissimilarity-result-locations', 'data'),
            prevent_initial_call=False
        )
        def update_map(active_tab, selected_location_id,
                      state_similarity_results, state_description_results,
                      change_similarity_results, change_description_results,
                      dissimilarity_results):
            """Update map with selected location and results from active tab."""
            from .. import context as app_ctx

            # Use results from the active tab
            result_location_ids = []

            if active_tab == 'state-similarity':
                result_location_ids = state_similarity_results or []
            elif active_tab == 'state-description':
                result_location_ids = state_description_results or []
            elif active_tab == 'change-similarity':
                result_location_ids = change_similarity_results or []
            elif active_tab == 'change-description':
                result_location_ids = change_description_results or []
            elif active_tab == 'dissimilarity':
                result_location_ids = dissimilarity_results or []

            fig = create_location_map(
                projects_df=app_ctx.PROJECTS_DF,
                selected_location_id=selected_location_id,
                result_location_ids=result_location_ids,
                center_lat=self.center_lat,
                center_lon=self.center_lon,
                zoom=self.zoom
            )
            return fig

    @property
    def layout(self) -> DashComponent:
        """Return the map layout."""
        return dcc.Graph(
            id='main-map',
            style={'height': self.height},
            config={'displayModeBar': False, 'scrollZoom': True}
        )
