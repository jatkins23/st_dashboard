"""Map-related callbacks."""

from dash import Input, Output

from ..utils.map_utils import create_location_map
from .. import state


def register_map_callbacks(app):
    """Register map callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('main-map', 'figure'),
        Input('query-location-id', 'data'),
        Input('result-locations', 'data'),
        prevent_initial_call=False
    )
    def update_map(query_location_id, result_location_ids):
        """Update map with selected location and results."""
        center_lat = 40.7128
        center_lon = -74.0060

        fig = create_location_map(
            # locations_df=state.ALL_LOCATIONS_DF,
            projects_df=state.PROJECTS_DF,
            selected_location_id=query_location_id,
            result_location_ids=result_location_ids or [],
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=11
        )

        return fig
