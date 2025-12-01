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
        Input('search-btn', 'n_clicks'),
        State('location-id-input', 'value'),
        State('year-dropdown', 'value'),
        State('target-year-dropdown', 'value'),
        State('limit-dropdown', 'value'),
        State('use-faiss-checkbox', 'value'),
        State('use-whitening-checkbox', 'value'),
        prevent_initial_call=True
    )
    def handle_search(n_clicks, location_id, year, target_year, limit, use_faiss, use_whitening):
        """Handle state search."""
        if not location_id or not year:
            return (
                dbc.Alert("Please enter location ID and year", color='warning'),
                {'display': 'block'},
                []
            )

        try:
            use_faiss_enabled = 'faiss' in (use_faiss or [])
            use_whitening_enabled = 'whitening' in (use_whitening or [])

            # Create and execute query
            query = ImageToImageStateQuery(
                config=state.CONFIG,
                db=state.DB,
                location_id=location_id,
                year=year,
                target_years=[target_year] if target_year else None,
                limit=limit,
                use_faiss=use_faiss_enabled,
                use_whitening=use_whitening_enabled,
                remove_self=True
            )

            results_set = query.search()

            if len(results_set) == 0:
                return (
                    dbc.Alert(f"No results found", color='info'),
                    {'display': 'block'},
                    []
                )

            # Enrich results with street names
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                for result in results_set:
                    result.enrich_street_names(con, state.CONFIG.universe_name)

            # Create results panel from results set
            results_panel = ResultsPanel(id_prefix='results', results=results_set)

            result_location_ids = [r.location_id for r in results_set]

            return (
                results_panel.content,
                {'display': 'block'},
                result_location_ids
            )

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                []
            )
