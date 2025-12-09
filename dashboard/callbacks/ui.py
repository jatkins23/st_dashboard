"""UI-related callbacks (form rendering, panel toggles)."""

from dash import Input, Output, State, html

import logging

from streettransformer.db.database import get_connection
from .. import state

logger = logging.getLogger(__name__)

def register_ui_callbacks(app):
    """Register UI callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('street-selector', 'data'),
        Input('street-selector', 'value'),
        State('street-selector', 'data'),
        prevent_initial_call=False
    )
    def filter_street_options(selected_streets, current_data):
        """Filter street options to only show valid combinations.

        When streets are selected, only show streets that appear in locations
        that also have ALL the currently selected streets.
        """
        # If no streets selected, return all streets
        if not selected_streets or len(selected_streets) == 0:
            return current_data

        try:
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                # Build conditions to find locations with ALL selected streets
                street_conditions = []
                for street in selected_streets:
                    street_conditions.append(f"""
                        (street1 = '{street}'
                         OR street2 = '{street}'
                         OR list_contains(additional_streets, '{street}'))
                    """)

                where_clause = " AND ".join(street_conditions)

                # Get all streets from locations that match the selected streets
                query = f"""
                    WITH matching_locations AS (
                        SELECT location_id
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE {where_clause}
                    ),
                    all_streets AS (
                        SELECT street1 as street
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE location_id IN (SELECT location_id FROM matching_locations)
                          AND street1 IS NOT NULL
                        UNION
                        SELECT street2 as street
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE location_id IN (SELECT location_id FROM matching_locations)
                          AND street2 IS NOT NULL
                        UNION
                        SELECT UNNEST(additional_streets) as street
                        FROM {state.CONFIG.universe_name}.locations
                        WHERE location_id IN (SELECT location_id FROM matching_locations)
                          AND additional_streets IS NOT NULL
                    )
                    SELECT DISTINCT street
                    FROM all_streets
                    ORDER BY street
                """

                df = con.execute(query).df()
                valid_streets = df['street'].dropna().tolist()

                # Return filtered options
                return [{"label": s, "value": s} for s in valid_streets]

        except Exception as e:
            logger.error(f"Error filtering street options: {e}", exc_info=True)
            # On error, return current data
            return current_data

    @app.callback(
        Output('results-collapse', 'is_open'),
        Output('results-collapse-btn', 'children'),
        Input('results-collapse-btn', 'n_clicks'),
        State('results-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_results_panel(n_clicks, is_open):
        """Toggle results panel collapse."""
        new_state = not is_open
        icon = html.I(className='fas fa-chevron-up' if new_state else 'fas fa-chevron-down')
        return new_state, icon

    @app.callback(
        Output('details-collapse', 'is_open'),
        Output('details-collapse-btn', 'children'),
        Input('details-collapse-btn', 'n_clicks'),
        State('details-collapse', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_details_panel(n_clicks, is_open):
        """Toggle details panel collapse."""
        new_state = not is_open
        icon = html.I(className='fas fa-chevron-up' if new_state else 'fas fa-chevron-down')
        return new_state, icon
