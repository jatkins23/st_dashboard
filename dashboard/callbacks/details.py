"""Location details callbacks."""

import logging
from pathlib import Path

from dash import Input, Output, html
import dash_bootstrap_components as dbc

from streettransformer.db.database import get_connection

from ..utils.display import encode_image_to_base64
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
        prevent_initial_call=False
    )
    def update_details(location_id, click_data):
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
                    ORDER BY year DESC
                    LIMIT 5
                """
                images_df = con.execute(image_query).df()

                # Create details display
                details = [
                    html.H6(f"Location {location_id}", className='fw-bold'),
                    html.P(street_name, className='text-muted'),
                    html.Hr()
                ]

                if not images_df.empty:
                    # Create carousel items
                    carousel_items = []
                    for _, img_row in images_df.iterrows():
                        img_path = Path(img_row['image_path'])
                        if img_path.exists():
                            img_base64 = encode_image_to_base64(img_path)
                            if img_base64:
                                carousel_items.append({
                                    'key': str(img_row['year']),
                                    'src': img_base64,
                                    'header': f"Year {img_row['year']}",
                                    'caption': ''
                                })

                    if carousel_items:
                        details.append(html.H6("Images:", className='mt-3 mb-2'))
                        details.append(
                            dbc.Carousel(
                                items=carousel_items,
                                controls=True,
                                indicators=True,
                                interval=None,
                                className='mb-3'
                            )
                        )

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
