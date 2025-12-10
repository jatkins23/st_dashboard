"""Main dashboard component."""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent

from .base import BaseComponent
from .results.results_panel import ResultsPanel
from .details.details_panel import DetailsPanel
from .search_form.state_search_form import StateSearchForm
from .search_form.change_search_form import ChangeSearchForm
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

        # Create search form instances for each tab
        self.state_search_form = StateSearchForm(
            available_years=available_years,
            all_streets=all_streets
        )
        self.change_search_form = ChangeSearchForm(
            available_years=available_years,
            all_streets=all_streets
        )

        # Shared components (used by all tabs)
        self.map_component = Map()
        self.results_panel = ResultsPanel()
        self.details_panel = DetailsPanel()


    def register_callbacks(self, app):
        """Register all component callbacks.

        This method registers callbacks for all child components plus
        the search callbacks for each tab type.
        """
        from dash import Input, Output, State
        import dash_bootstrap_components as dbc

        # Register child component callbacks
        self.state_search_form.register_callbacks(app)
        self.change_search_form.register_callbacks(app)
        self.map_component.register_callbacks(app)
        self.results_panel.register_callbacks(app)
        self.details_panel.register_callbacks(app)

        # Track active tab and clear street selectors when switching
        @app.callback(
            Output('active-search-tab', 'data'),
            Output('state-search-form--street-selector', 'value'),
            Output('change-search-form--street-selector', 'value'),
            Input('search-tabs', 'value')
        )
        def track_active_tab(tab_value):
            """Track which search tab is currently active and clear street selectors."""
            return tab_value, [], []

        # Register STATE search callback
        @app.callback(
            Output('results-content', 'children'),
            Output('results-card', 'style'),
            Output('state-result-locations', 'data'),
            Output('state-query-params', 'data'),
            Input('state-search-form--search-btn', 'n_clicks'),
            State('selected-location-id', 'data'),
            State('state-search-form--year-selector', 'value'),
            State('state-search-form--target-year-selector', 'value'),
            State('state-search-form--limit-dropdown', 'value'),
            State('state-search-form--media-type-selector', 'value'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_state_search(n_clicks, location_id, year, target_year, limit, media_type, active_tab):
            """Handle state search (image-to-image by year)."""
            from .. import state as app_state

            # Only process if state tab is active
            if active_tab != 'state':
                return dbc.Alert("Please switch to State Search tab", color='warning'), {'display': 'block'}, [], None

            if not location_id or not year:
                return (
                    dbc.Alert("Please select a location and year", color='warning'),
                    {'display': 'block'},
                    [],
                    None
                )

            try:
                results_set = self.state_search_form.execute_search(
                    state=app_state,
                    location_id=location_id,
                    year=year,
                    target_year=target_year,
                    limit=limit,
                    media_type=media_type
                )

                if len(results_set) == 0:
                    return (
                        dbc.Alert(f"No results found", color='info'),
                        {'display': 'block'},
                        [],
                        {'year': year}
                    )

                # Create results panel from results set
                results_panel = ResultsPanel(id_prefix='results', results=results_set)
                result_location_ids = [r.location_id for r in results_set]

                return (
                    results_panel.content,
                    {'display': 'block'},
                    result_location_ids,
                    {'year': year, 'target_year': target_year}
                )

            except Exception as e:
                logger.error(f"State search error: {e}", exc_info=True)
                return (
                    dbc.Alert(f"Error: {str(e)}", color='danger'),
                    {'display': 'block'},
                    [],
                    None
                )

        # Register CHANGE search callback
        @app.callback(
            Output('results-content', 'children', allow_duplicate=True),
            Output('results-card', 'style', allow_duplicate=True),
            Output('change-result-locations', 'data'),
            Output('change-query-params', 'data'),
            Input('change-search-form--search-btn', 'n_clicks'),
            State('selected-location-id', 'data'),
            State('change-search-form--year-from-selector', 'value'),
            State('change-search-form--year-to-selector', 'value'),
            State('change-search-form--limit-dropdown', 'value'),
            State('change-search-form--media-type-selector', 'value'),
            State('change-search-form--sequential-checkbox', 'value'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_change_search(n_clicks, location_id, year_from, year_to, limit, media_type, sequential_value, active_tab):
            """Handle change search (temporal change detection)."""
            from .. import state as app_state

            # Only process if change tab is active
            if active_tab != 'change':
                return dbc.Alert("Please switch to Change Search tab", color='warning'), {'display': 'block'}, [], None

            if not location_id or not year_from or not year_to:
                return (
                    dbc.Alert("Please select a location, from year, and to year", color='warning'),
                    {'display': 'block'},
                    [],
                    None
                )

            # Convert sequential checkbox value to boolean
            sequential = 'sequential' in (sequential_value or [])

            try:
                results_set = self.change_search_form.execute_search(
                    state=app_state,
                    location_id=location_id,
                    year_from=year_from,
                    year_to=year_to,
                    limit=limit,
                    media_type=media_type,
                    sequential=sequential
                )

                if len(results_set) == 0:
                    return (
                        dbc.Alert(f"No results found", color='info'),
                        {'display': 'block'},
                        [],
                        {'year_from': year_from, 'year_to': year_to}
                    )

                # Create results panel from results set
                results_panel = ResultsPanel(id_prefix='results', results=results_set)
                result_location_ids = [r.location_id for r in results_set]

                return (
                    results_panel.content,
                    {'display': 'block'},
                    result_location_ids,
                    {'year_from': year_from, 'year_to': year_to, 'sequential': sequential}
                )

            except Exception as e:
                logger.error(f"Change search error: {e}", exc_info=True)
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
        """Return the complete dashboard layout with tabs."""

        components = [
            # Header
            self._header(),

            # Tabs with different search forms
            dcc.Tabs(
                id='search-tabs',
                value='state',  # Default to state search tab
                children=[
                    dcc.Tab(
                        label='State Search',
                        value='state',
                        children=[
                            html.Div([
                                self.state_search_form.layout
                            ], style={'marginTop': '10px'})
                        ]
                    ),
                    dcc.Tab(
                        label='Change Search',
                        value='change',
                        children=[
                            html.Div([
                                self.change_search_form.layout
                            ], style={'marginTop': '10px'})
                        ]
                    ),
                ],
                className='mb-3'
            ),

            # Map with floating panels (shared across all tabs)
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
            dcc.Store(id='active-search-tab'),  # Track which tab is active
            dcc.Store(id='selected-location-id'),  # Shared across tabs

            # State search stores
            dcc.Store(id='state-result-locations'),
            dcc.Store(id='state-query-params'),

            # Change search stores
            dcc.Store(id='change-result-locations'),
            dcc.Store(id='change-query-params'),
        ]

        return dbc.Container(dmc.MantineProvider(components), fluid=True)
