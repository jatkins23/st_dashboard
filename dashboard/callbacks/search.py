"""Search-related callbacks."""

import logging

from dash import Input, Output, State
import dash_bootstrap_components as dbc

from streettransformer.db.database import get_connection
from streettransformer.query.queries.ask import ImageToImageStateQuery

from ..frontend.components.results.results_panel import ResultsPanel
from .. import state

logger = logging.getLogger(__name__)


def register_search_callbacks(app):
    """Register search callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('results-content', 'children'),
        Output('results-card', 'style'),
        Output('result-locations', 'data'),
        Output('query-year', 'data'),
        Input('search-btn', 'n_clicks'),
        State('location-id-input', 'value'),
        State('year-dropdown', 'value'),
        State('target-year-dropdown', 'value'),
        State('limit-dropdown', 'value'),
<<<<<<< HEAD
        prevent_initial_call=True
    )
    def handle_search(n_clicks, location_id, year, target_year, limit):
=======
        State('media-type-checkbox', 'value'),
        State('use-faiss-checkbox', 'value'),
        State('use-whitening-checkbox', 'value'),
        prevent_initial_call=True
    )
    def handle_search(n_clicks, location_id, year, target_year, limit, media_type, use_faiss, use_whitening):
>>>>>>> a6597d193907f1f57399b3aaec93a584cdca3f7a
        """Handle state search."""
        if not location_id or not year:
            return (
                dbc.Alert("Please enter location ID and year", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        try:
            # Configuration settings
            use_faiss_enabled = True
            use_whitening_enabled = False

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
                use_faiss=use_faiss_enabled,
                use_whitening=use_whitening_enabled,
                remove_self=True
            )

            results_set = query.search()

            if len(results_set) == 0:
                return (
                    dbc.Alert(f"No results found", color='info'),
                    {'display': 'block'},
                    [],
                    year
                )

            # Enrich results with street names and image paths
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                for result in results_set:
                    result.enrich_street_names(con, state.CONFIG.universe_name)
                    result.enrich_image_path(con, state.CONFIG.universe_name, selected_media_type)

            # Create results panel from results set
            results_panel = ResultsPanel(id_prefix='results', results=results_set)

            result_location_ids = [r.location_id for r in results_set]

            return (
                results_panel.content,
                {'display': 'block'},
                result_location_ids,
                year
            )

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                [],
                None
            )
