"""Map component for displaying locations and projects."""

from dash import dcc, Input, Output
from dash.development.base_component import Component as DashComponent

from .base import BaseComponent
from ..utils.map_utils import create_location_map
from .. import state

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
        - A query location is selected
        - Search results are returned
        """

        @app.callback(
            Output('main-map', 'figure'),
            Input('selected-location-id', 'data'),
            Input('result-locations', 'data'),
            prevent_initial_call=False
        )
        def update_map(query_location_id, result_location_ids):
            """Update map with selected location and results."""
            fig = create_location_map(
                projects_df=state.PROJECTS_DF,
                selected_location_id=query_location_id,
                result_location_ids=result_location_ids or [],
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
