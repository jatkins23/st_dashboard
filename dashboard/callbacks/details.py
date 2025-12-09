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
        Input('street-selector', 'value'),
        Input('main-map', 'clickData'),
        Input('query-year', 'data'),
        prevent_initial_call=False
    )
    def update_details(selected_streets, click_data, query_year):
        """Update details panel when location is selected."""
        location_id = None

        # Handle map click first (takes priority)
        if click_data and 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            if 'customdata' in point:
                location_id = point['customdata']

        # If no map click, handle street selection
        elif selected_streets and len(selected_streets) > 0:
            try:
                with get_connection(state.CONFIG.database_path, read_only=True) as con:
                    # Find locations that match ALL selected streets
                    street_conditions = []
                    for street in selected_streets:
                        street_conditions.append(f"""
                            (street1 = '{street}'
                             OR street2 = '{street}'
                             OR list_contains(additional_streets, '{street}'))
                        """)

                    where_clause = " AND ".join(street_conditions)

                    query = f"""
                        SELECT location_id
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE {where_clause}
                        LIMIT 1
                    """
                    result = con.execute(query).df()

                    if not result.empty:
                        location_id = result.iloc[0]['location_id']
            except Exception as e:
                logger.error(f"Error finding location by streets: {e}", exc_info=True)

        if not location_id:
            return (
                html.Div("Select streets or click on the map", className='text-muted fst-italic'),
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
                    SELECT path, media_type, year
                    FROM {state.CONFIG.universe_name}.media_embeddings
                    WHERE location_id = {location_id}
                        AND media_type = 'image'
                        AND path IS NOT NULL
                    ORDER BY year ASC
                    LIMIT 5
                """
                images_df = con.execute(image_query).df()

                # Rename 'path' to 'image_path' for backward compatibility with viewer
                if not images_df.empty:
                    images_df = images_df.rename(columns={'path': 'image_path'})

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
