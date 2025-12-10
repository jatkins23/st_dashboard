"""Main dashboard component."""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent

from .base import BaseComponent
from .results import ResultsPanel
from .details import DetailsPanel
from .search_form import (
    ImageStateSearchForm,
    ImageChangeSearchForm,
    TextStateSearchForm,
    TextChangeSearchForm
)
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
        id_prefix: str = 'dashboard',
        enable_image_search: bool = True,
        enable_text_search: bool = True
    ):
        """Initialize the dashboard.

        Args:
            universe_name: Name of the universe being explored
            available_years: List of available years for the search form
            all_streets: List of all unique street names
            id_prefix: Prefix for component IDs
            enable_image_search: Whether to enable image-based search (default: True)
            enable_text_search: Whether to enable text-based search (default: False)
        """
        super().__init__(id_prefix=id_prefix)
        self.universe_name = universe_name
        self.available_years = available_years
        self.all_streets = all_streets
        self.enable_image_search = enable_image_search
        self.enable_text_search = enable_text_search

        # Create search form instances for Image search tabs
        if self.enable_image_search:
            self.image_state_search_form = ImageStateSearchForm(
                available_years=available_years,
                all_streets=all_streets
            )
            self.image_change_search_form = ImageChangeSearchForm(
                available_years=available_years,
                all_streets=all_streets
            )
        else:
            self.image_state_search_form = None
            self.image_change_search_form = None

        # Create search form instances for Text search tabs
        if self.enable_text_search:
            self.text_state_search_form = TextStateSearchForm(
                available_years=available_years,
                all_streets=all_streets
            )
            self.text_change_search_form = TextChangeSearchForm(
                available_years=available_years,
                all_streets=all_streets
            )
        else:
            self.text_state_search_form = None
            self.text_change_search_form = None

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
        if self.enable_image_search:
            self.image_state_search_form.register_callbacks(app)
            self.image_change_search_form.register_callbacks(app)
        if self.enable_text_search:
            self.text_state_search_form.register_callbacks(app)
            self.text_change_search_form.register_callbacks(app)

        self.map_component.register_callbacks(app)
        self.results_panel.register_callbacks(app)
        self.details_panel.register_callbacks(app)

        # Track active search type and sub-tab, clear inputs when switching
        outputs = [Output('active-search-tab', 'data')]
        inputs = []

        # Add outputs for clearing image search inputs
        if self.enable_image_search:
            outputs.extend([
                Output('image-state-search-form--street-selector', 'value'),
                Output('image-change-search-form--street-selector', 'value')
            ])
            inputs.append(Input('image-search-tabs', 'value'))

        # Add outputs for clearing text search inputs
        if self.enable_text_search:
            outputs.extend([
                Output('text-state-search-form--text-input', 'value'),
                Output('text-change-search-form--text-input', 'value')
            ])
            inputs.append(Input('text-search-tabs', 'value'))

        # Add top-level search type tabs input
        inputs.insert(0, Input('search-tabs', 'value'))

        @app.callback(*outputs, *inputs)
        def track_active_tab(*args):
            """Track which search tab is currently active and clear inputs when switching."""
            search_type = args[0]  # 'image' or 'text'

            # Determine the active sub-tab
            if self.enable_image_search and search_type == 'image':
                sub_tab = args[1] if len(args) > 1 else 'image-state'
                active_tab = sub_tab
            elif self.enable_text_search and search_type == 'text':
                # Text tab index depends on whether image search is enabled
                idx = 1 if self.enable_image_search else 1
                sub_tab = args[idx] if len(args) > idx else 'text-state'
                active_tab = sub_tab
            else:
                active_tab = 'image-state'  # Default

            # Build return values: active_tab + cleared inputs
            result = [active_tab]

            # Clear image search inputs
            if self.enable_image_search:
                result.extend([[], []])  # Clear both street selectors

            # Clear text search inputs
            if self.enable_text_search:
                result.extend(['', ''])  # Clear both text inputs

            return tuple(result) if len(result) > 1 else result[0]

        # Register IMAGE STATE search callback
        if self.enable_image_search:
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
            def handle_image_state_search(n_clicks, location_id, year, target_year, limit, media_type, active_tab):
                image_state_search(n_clicks, location_id, year, target_year, limit, media_type, active_tab)

        # Register IMAGE CHANGE search callback
        if self.enable_image_search:
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
            def handle_image_change_search(n_clicks, location_id, year_from, year_to, limit, media_type, sequential_value, active_tab):
                """Handle change search (temporal change detection)."""
                self.image_change_search(n_clicks, location_id, year_from, year_to, limit, media_type, sequential_value, active_tab)

        # Register TEXT STATE search callback
        if self.enable_text_search:
            @app.callback(
                Output('results-content', 'children', allow_duplicate=True),
                Output('results-card', 'style', allow_duplicate=True),
                Output('text-state-result-locations', 'data'),
                Output('text-state-query-params', 'data'),
                Input('state-text-search-form--search-btn', 'n_clicks'),
                State('state-text-search-form--text-input', 'value'),
                State('state-text-search-form--year-selector', 'value'),
                State('state-text-search-form--target-year-selector', 'value'),
                State('state-text-search-form--limit-dropdown', 'value'),
                State('state-text-search-form--media-type-selector', 'value'),
                State('active-search-tab', 'data'),
                prevent_initial_call=True
            )
            def handle_text_state_search(n_clicks, text, year, target_year, limit, media_type, active_tab):
                self.text_state_search(n_clicks, text, year, target_year, limit, media_type, active_tab)
                
        # Register TEXT CHANGE search callback
        if self.enable_text_search:
            @app.callback(
                Output('results-content', 'children', allow_duplicate=True),
                Output('results-card', 'style', allow_duplicate=True),
                Output('text-change-result-locations', 'data'),
                Output('text-change-query-params', 'data'),
                Input('change-text-search-form--search-btn', 'n_clicks'),
                State('change-text-search-form--text-input', 'value'),
                State('change-text-search-form--year-from-selector', 'value'),
                State('change-text-search-form--year-to-selector', 'value'),
                State('change-text-search-form--limit-dropdown', 'value'),
                State('change-text-search-form--media-type-selector', 'value'),
                State('change-text-search-form--sequential-checkbox', 'value'),
                State('active-search-tab', 'data'),
                prevent_initial_call=True
            )
            def handle_text_change_search(n_clicks, text, year_from, year_to, limit, media_type, sequential_value, active_tab):
                self.text_change_search(n_clicks, text, year_from, year_to, limit, media_type, sequential_value, active_tab)

    def _header(self):
        return dbc.Row([
            dbc.Col([
                html.H3(
                    f"Street Transformer - {self.universe_name.upper()}",
                    className='text-light mb-3'
                )
            ])
        ], className='mt-3')

    def _top_tab_layout(self, type:str) -> DashComponent:
        # TODO: Maybe swap type and mode? Or find better names for them
        def _child_tab(mode) -> DashComponent:
            print('here')
            search_form = getattr(self, f'{type}_{mode}_search_form')
            if not search_form:
                logger.error(f'Search form not found: {type}, {mode}')
                return None
            
            return dcc.Tab(
                label=f'{mode.title()} Search',
                value=f'{type}-{mode}',
                children=[
                    html.Div([
                        search_form.layout
                    ], style={'marginTop': '10px'})
                ]
            )
        
        return dcc.Tab(
            label=f'{type.title()}-Based Search',
            value=f'{type}-tabs',
            children=dcc.Tabs(
                id=f'{type}-search-tabs',
                value=f'{type}-state',  # Default to state search tab,
                children = [_child_tab(mode) for mode in ['state','change']]
            )
        )

    @property
    def layout(self) -> DashComponent:
        """Return the complete dashboard layout with tabs."""
        
        how_many_tabs = self.enable_image_search + self.enable_text_search
        if how_many_tabs > 1:
            # TODO: Only use top_tabs if more than 1
            #top_tabs = dcc.Tab()
            pass
        
        top_tabs = []
        
        if self.enable_image_search:
            top_tabs.append(self._top_tab_layout('image'))
            
        if self.enable_text_search:
            top_tabs.append(self._top_tab_layout('text'))
        
        components = [
            # Header
            self._header(),

            # Tabs with different search forms
            dcc.Tabs(
                id='search-tabs',
                value='image-tabs',  # Default to first enabled search type
                children=top_tabs,
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

            # Image state search stores
            dcc.Store(id='state-result-locations'),
            dcc.Store(id='state-query-params'),

            # Image change search stores
            dcc.Store(id='change-result-locations'),
            dcc.Store(id='change-query-params'),

            # Text state search stores
            dcc.Store(id='text-state-result-locations'),
            dcc.Store(id='text-state-query-params'),

            # Text change search stores
            dcc.Store(id='text-change-result-locations'),
            dcc.Store(id='text-change-query-params'),
        ]

        return dbc.Container(
            dmc.MantineProvider(
                components,
                theme={
                    "colorScheme": "dark",
                    "primaryColor": "blue"
                }
            ),
            fluid=True
        )


    # ------- Search handlers  ------- #
    def handle_image_state_search(self, n_clicks, location_id, year, target_year, limit, media_type, active_tab):
        from .. import state as app_state
        
        """Handle state search (image-to-image by year)."""
        # Only process if state tab is active
        if active_tab != 'image-state':
            return dbc.Alert("Please switch to State Search tab", color='warning'), {'display': 'block'}, [], None

        if not location_id or not year:
            return (
                dbc.Alert("Please select a location and year", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        try:
            results_set = self.image_state_search_form.execute_search(
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
        
    def image_change_search(self, n_clicks, location_id, year_from, year_to, sequential_value, limit, media_type, active_tab):
        from .. import state as app_state
        
        # Only process if change tab is active
        if active_tab != 'image-change':
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
            results_set = self.image_change_search_form.execute_search(
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
    
    def text_state_search(self, n_clicks, text, year, target_year, limit, media_type, active_tab):
        """Handle text state search (text-to-image by year)."""
        from .. import state as app_state

        # Only process if text-state tab is active
        if active_tab != 'text-state':
            return dbc.Alert("Please switch to Text State Search tab", color='warning'), {'display': 'block'}, [], None

        if not text or not year:
            return (
                dbc.Alert("Please enter text and select a year", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        try:
            results_set = self.text_state_search_form.execute_search(
                state=app_state,
                text=text,
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
                {'year': year, 'target_year': target_year, 'text': text}
            )

        except Exception as e:
            logger.error(f"Text state search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                [],
                None
            )

    def text_change_search(self, n_clicks, text, year_from, year_to, limit, media_type, sequential_value, active_tab):
        """Handle text change search (text-to-image temporal change detection)."""
        from .. import state as app_state

        # Only process if text-change tab is active
        if active_tab != 'text-change':
            return dbc.Alert("Please switch to Text Change Search tab", color='warning'), {'display': 'block'}, [], None

        if not text or not year_from or not year_to:
            return (
                dbc.Alert("Please enter text, from year, and to year", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        # Convert sequential checkbox value to boolean
        sequential = 'sequential' in (sequential_value or [])

        try:
            results_set = self.text_change_search_form.execute_search(
                state=app_state,
                text=text,
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
                {'year_from': year_from, 'year_to': year_to, 'sequential': sequential, 'text': text}
            )

        except Exception as e:
            logger.error(f"Text change search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                [],
                None
            )
        