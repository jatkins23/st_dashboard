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

    def register_callbacks(self, app):
        """Register callbacks for the dashboard.

        Note: Callbacks are registered separately in the callbacks module.
        This method is here for consistency with BaseComponent interface.
        """
        pass

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
                    SearchForm(
                        title = "Image-to-Image State Search",
                        available_years=self.available_years,
                        all_streets=self.all_streets
                    ).layout
                ])
            ]),

            # Map with floating panels
            dbc.Row([
                dbc.Col([
                    html.Div([
                        # Map
                        Map().layout,
                        # Floating results panel
                        ResultsPanel()(),
                        # Floating details panel
                        DetailsPanel()()
                    ], style={'position': 'relative'})
                ]),
            ], className='mb-3'),

            # Data stores
            dcc.Store(id='query-location-id'),
            dcc.Store(id='query-year'),
            dcc.Store(id='result-locations'),
        ]

        return dbc.Container(dmc.MantineProvider(components), fluid=True)
