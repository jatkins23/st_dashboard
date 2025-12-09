"""Main dashboard component."""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent

from .base import BaseComponent
from .results.results_panel import ResultsPanel
from .details.details_panel import DetailsPanel
from .search_form import SearchForm
from .map_component import Map

import logging
logger = logging.getLogger(__name__)


class Dashboard(BaseComponent):
    """Main dashboard component that holds all app data and layout.

    This component encapsulates the entire dashboard layout including
    search form, map, results panel, and details panel.
    """

    def __init__(
        self,
        universe_name: str,
        available_years: list,
        all_streets: list,
        id_prefix: str = 'dashboard'
    ):
        """Initialize the dashboard.

        Args:
            universe_name: Name of the universe being explored
            available_years: List of available years for the search form
            all_streets: List of all unique street names
            id_prefix: Prefix for component IDs
        """
        super().__init__(id_prefix=id_prefix)
        self.universe_name = universe_name
        self.available_years = available_years
        self.all_streets = all_streets

        # Create component instances
        self.search_form = SearchForm(
            available_years=available_years,
            all_streets=all_streets
        )
        self.map_component = Map()
        self.results_panel = ResultsPanel()
        self.details_panel = DetailsPanel()


    def register_callbacks(self, app):
        """Register all component callbacks.

        This method registers callbacks for all child components plus
        the main search callback that coordinates between them.
        """
        from dash import Input, Output, State
        import dash_bootstrap_components as dbc

        # Register child component callbacks
        self.search_form.register_callbacks(app)
        self.map_component.register_callbacks(app)
        self.results_panel.register_callbacks(app)
        self.details_panel.register_callbacks(app)

        # Register search callback (coordinates SearchForm → ResultsPanel)
        @app.callback(
            Output('results-content', 'children'),
            Output('results-card', 'style'),
            Output('result-locations', 'data'),
            Output('query-year', 'data'),
            Input('search-btn', 'n_clicks'),
            State('street-selector', 'value'),
            State('year-selector', 'value'),
            State('target-year-selector', 'value'),
            State('limit-dropdown', 'value'),
            State('media-type-selector', 'value'),
            State('use-faiss-checkbox', 'value'),
            State('use-whitening-checkbox', 'value'),
            prevent_initial_call=True
        )
        def handle_search(n_clicks, selected_streets, year, target_year, limit, media_type, use_faiss, use_whitening):
            """Handle state search."""
            from .search import get_location_from_streets, execute_image_search
            from .. import state

            # Get location_id from selected streets
            location_id = get_location_from_streets(selected_streets, state)

            if not location_id or not year:
                return (
                    dbc.Alert("Please select streets and year", color='warning'),
                    {'display': 'block'},
                    [],
                    None
                )

            try:
                results_set = execute_image_search(location_id, year, target_year, limit, media_type, state)

                if len(results_set) == 0:
                    return (
                        dbc.Alert(f"No results found", color='info'),
                        {'display': 'block'},
                        [],
                        year
                    )

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


    def _header(self):
        return dbc.Row([
            dbc.Col([
                html.H3(
                    f"Street Transformer - {self.universe_name.upper()}",
                    className='text-light mb-3'
                )
            ])
        ], className='mt-3')

    @property
    def layout(self) -> DashComponent:
        """Return the complete dashboard layout."""

        components = [
            # Header
            self._header(),

            # Search card
            dbc.Row([
                dbc.Col([
                    self.search_form.layout
                ])
            ]),

            # Map with floating panels
            dbc.Row([
                dbc.Col([
                    html.Div([
                        # Map
                        self.map_component.layout,
                        # Floating results panel
                        self.results_panel(),
                        # Floating details panel
                        self.details_panel()
                    ], style={'position': 'relative'})
                ]),
            ], className='mb-3'),

            # Data stores
            dcc.Store(id='query-location-id'),
            dcc.Store(id='query-year'),
            dcc.Store(id='result-locations'),
        ]

        return dbc.Container(dmc.MantineProvider(components), fluid=True)
