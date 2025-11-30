"""UI-related callbacks (form rendering, panel toggles)."""

from dash import Input, Output, State, html

from ..frontend.layout import create_search_form
from .. import state


def register_ui_callbacks(app):
    """Register UI callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('search-form', 'children'),
        Input('main-map', 'id')  # Dummy input to trigger on load
    )
    def render_search_form(_):
        """Render the search form."""
        return create_search_form(state.AVAILABLE_YEARS)

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
