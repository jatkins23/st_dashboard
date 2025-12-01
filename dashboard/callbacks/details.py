"""Location details callbacks."""

import logging

from dash import Input, Output, html
import dash_bootstrap_components as dbc

from streettransformer.db.database import get_connection

from ..frontend.components.details import (
    DetailsStatsViewer,
    DetailsImageViewer,
    DetailsDocumentViewer,
    DetailsProjectViewer
)
from .. import state

logger = logging.getLogger(__name__)


def register_details_callbacks(app):
    """Register details callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('details-content', 'children'),
        Output('details-card', 'style'),
        Output('query-location-id', 'data'),
        Input('location-id-input', 'value'),
        Input('main-map', 'clickData'),
        Input('query-year', 'data'),
        prevent_initial_call=False
    )
    def update_details(location_id, click_data, query_year):
        """Update details panel when location is selected."""
        # Handle map click
        if click_data and 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            if 'customdata' in point:
                location_id = point['customdata']

        if not location_id:
            return (
                html.Div("Enter a location ID or click on the map", className='text-muted fst-italic'),
                {'display': 'none'},
                None
            )

        try:
            # Get location info
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                query = f"""
                    SELECT
                        location_id,
                        COALESCE(
                            array_to_string(additional_streets, ', '),
                            CONCAT(street1, ' & ', street2)
                        ) as street_name
                    FROM {state.CONFIG.universe_name}.locations
                    WHERE location_id = {location_id}
                """
                result = con.execute(query).df()

                if result.empty:
                    return (
                        dbc.Alert(f"Location {location_id} not found", color='warning'),
                        {'display': 'block'},
                        location_id
                    )

                street_name = result.iloc[0]['street_name']

                # Get images
                image_query = f"""
                    SELECT image_path, year
                    FROM {state.CONFIG.universe_name}.image_embeddings
                    WHERE location_id = {location_id}
                        AND image_path IS NOT NULL
                    ORDER BY year ASC
                    LIMIT 5
                """
                images_df = con.execute(image_query).df()

                # Create viewer instances with data and combine their content
                details = []

                # 1. Stats/Header Section
                stats_viewer = DetailsStatsViewer(location_id=location_id, street_name=street_name)
                details.extend(stats_viewer.content)

                # 2. Image Carousel Section
                image_viewer = DetailsImageViewer(images_df=images_df, query_year=query_year)
                details.extend(image_viewer.content)

                # 3. Document Section (stub)
                document_viewer = DetailsDocumentViewer(location_id=location_id)
                details.extend(document_viewer.content)

                # 4. Project Section (stub)
                project_viewer = DetailsProjectViewer(location_id=location_id)
                details.extend(project_viewer.content)

                return (
                    html.Div(details),
                    {'display': 'block'},
                    location_id
                )

        except Exception as e:
            logger.error(f"Error getting location details: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                location_id
            )
