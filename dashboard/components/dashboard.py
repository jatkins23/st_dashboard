"""Main dashboard component."""

from dash import dcc, html, Input, Output, State
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
    TextChangeSearchForm,
    DissimilaritySearchForm
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
        all_boroughs: list = None,
        id_prefix: str = 'dashboard'
    ):
        """Initialize the dashboard with all search forms.

        Args:
            universe_name: Name of the universe being explored
            available_years: List of available years for the search form
            all_streets: List of all unique street names
            all_boroughs: List of all unique borough names
            id_prefix: Prefix for component IDs
        """
        super().__init__(id_prefix=id_prefix)
        self.universe_name = universe_name
        self.available_years = available_years
        self.all_streets = all_streets
        self.all_boroughs = all_boroughs or []

        # Create all 5 search form instances explicitly
        self.state_similarity_form = ImageStateSearchForm(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs
        )

        self.dissimilarity_form = DissimilaritySearchForm(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs
        )

        self.state_description_form = TextStateSearchForm(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs
        )

        self.change_similarity_form = ImageChangeSearchForm(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs
        )

        self.change_description_form = TextChangeSearchForm(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs
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
        # Register all search form callbacks
        self.state_similarity_form.register_callbacks(app)
        self.dissimilarity_form.register_callbacks(app)
        self.state_description_form.register_callbacks(app)
        self.change_similarity_form.register_callbacks(app)
        self.change_description_form.register_callbacks(app)

        # Register shared component callbacks
        self.map_component.register_callbacks(app)
        self.results_panel.register_callbacks(app)
        self.details_panel.register_callbacks(app)

        # Track active search tab (nested structure)
        @app.callback(
            Output('active-search-tab', 'data'),
            Input('search-tabs', 'value')
        )
        def track_active_tab(tab_value):
            """Track which search tab is currently active.

            Other components (map, details panel) use this to react to tab changes.
            """
            return tab_value

        # Show/hide map container based on master tab
        @app.callback(
            Output('search-map-container', 'style'),
            Input('master-tabs', 'value')
        )
        def toggle_map_visibility(master_tab_value):
            """Show map only when Search master tab is active."""
            if master_tab_value == 'master-search':
                return {'display': 'block'}
            else:
                return {'display': 'none'}

        # Register IMAGE STATE search callback
        @app.callback(
            Output('results-content', 'children'),
            Output('results-card', 'style'),
            Output('state-similarity-result-locations', 'data'),
            Output('state-similarity-query-params', 'data'),
            Input('state-search-form--search-btn', 'n_clicks'),
            State('selected-location-id', 'data'),
            State('state-search-form--borough-selector', 'value'),
            State('state-search-form--year-selector', 'value'),
            State('state-search-form--target-year-selector', 'value'),
            State('state-search-form--limit-dropdown', 'value'),
            State('state-search-form--media-type-selector', 'value'),
            State('state-search-form--use-faiss-checkbox', 'checked'),
            State('state-search-form--use-whitening-checkbox', 'checked'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_image_state_search(n_clicks, location_id, boroughs, year, target_year, limit, media_type, use_faiss, use_whitening, active_tab):
            return self.image_state_search(n_clicks, location_id, boroughs, year, target_year, limit, media_type, use_faiss, use_whitening, active_tab)

        # Register IMAGE CHANGE search callback
        @app.callback(
            Output('results-content', 'children', allow_duplicate=True),
            Output('results-card', 'style', allow_duplicate=True),
            Output('change-similarity-result-locations', 'data'),
            Output('change-similarity-query-params', 'data'),
            Input('change-search-form--search-btn', 'n_clicks'),
            State('selected-location-id', 'data'),
            State('change-search-form--borough-selector', 'value'),
            State('change-search-form--year-from-selector', 'value'),
            State('change-search-form--year-to-selector', 'value'),
            State('change-search-form--limit-dropdown', 'value'),
            State('change-search-form--media-type-selector', 'value'),
            State('change-search-form--sequential-checkbox', 'value'),
            State('change-search-form--use-faiss-checkbox', 'checked'),
            State('change-search-form--use-whitening-checkbox', 'checked'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_image_change_search(n_clicks, location_id, boroughs, year_from, year_to, limit, media_type, sequential_value, use_faiss, use_whitening, active_tab):
            return self.image_change_search(n_clicks, location_id, boroughs, year_from, year_to, limit, media_type, sequential_value, use_faiss, use_whitening, active_tab)

        # Register TEXT STATE search callback
        @app.callback(
            Output('results-content', 'children', allow_duplicate=True),
            Output('results-card', 'style', allow_duplicate=True),
            Output('state-description-result-locations', 'data'),
            Output('state-description-query-params', 'data'),
            Input('state-text-search-form--search-btn', 'n_clicks'),
            State('state-text-search-form--text-input', 'value'),
            State('state-text-search-form--borough-selector', 'value'),
            State('state-text-search-form--target-year-selector', 'value'),
            State('state-text-search-form--limit-dropdown', 'value'),
            State('state-text-search-form--media-type-selector', 'value'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_text_state_search(n_clicks, text, boroughs, target_year, limit, media_type, active_tab):
            return self.text_state_search(n_clicks, text, boroughs, target_year, limit, media_type, active_tab)

        # Register TEXT CHANGE search callback
        @app.callback(
            Output('results-content', 'children', allow_duplicate=True),
            Output('results-card', 'style', allow_duplicate=True),
            Output('change-description-result-locations', 'data'),
            Output('change-description-query-params', 'data'),
            Input('change-text-search-form--search-btn', 'n_clicks'),
            State('change-text-search-form--text-input', 'value'),
            State('change-text-search-form--borough-selector', 'value'),
            State('change-text-search-form--limit-dropdown', 'value'),
            State('change-text-search-form--media-type-selector', 'value'),
            State('change-text-search-form--sequential-checkbox', 'value'),
            State('active-search-tab', 'data'),
            prevent_initial_call=True
        )
        def handle_text_change_search(n_clicks, text, boroughs, limit, media_type, sequential_value, active_tab):
            return self.text_change_search(n_clicks, text, boroughs, limit, media_type, sequential_value, active_tab)

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
        """Return the complete dashboard layout with master tab structure."""
        components = [
            # Header
            self._header(),

            # Master tabs: Search, Data, Paper, About, Help
            dmc.Tabs(
                [
                    # Master tab list
                    dmc.TabsList(
                        [
                            dmc.TabsTab('Search', value='master-search'),
                            dmc.TabsTab('Data', value='master-data'),
                            dmc.TabsTab('Paper', value='master-paper'),
                            dmc.TabsTab('About', value='master-about'),
                            dmc.TabsTab('Help', value='master-help'),
                        ],
                        style={
                            'marginBottom': '20px',
                            'borderBottom': '3px solid #dee2e6',
                            'gap': '12px'
                        }
                    ),

                    # Master Tab Panel 1: Search (contains the 5 search tabs)
                    dmc.TabsPanel(
                        dmc.Tabs(
                            [
                                dmc.TabsList(
                                    [
                                        dmc.TabsTab('State Similarity', value='state-similarity'),
                                        dmc.TabsTab('State Dissimilarity', value='dissimilarity'),
                                        dmc.TabsTab('State Description', value='state-description'),
                                        dmc.TabsTab('Change Similarity', value='change-similarity'),
                                        dmc.TabsTab('Change Description', value='change-description'),
                                    ],
                                    style={
                                        'marginBottom': '15px',
                                        'borderBottom': '2px solid #495057',
                                        'gap': '8px'
                                    }
                                ),
                                # Sub-tab panels with search forms
                                dmc.TabsPanel(self.state_similarity_form.layout, value='state-similarity'),
                                dmc.TabsPanel(self.dissimilarity_form.layout, value='dissimilarity'),
                                dmc.TabsPanel(self.state_description_form.layout, value='state-description'),
                                dmc.TabsPanel(self.change_similarity_form.layout, value='change-similarity'),
                                dmc.TabsPanel(self.change_description_form.layout, value='change-description'),
                            ],
                            id='search-tabs',
                            value='state-similarity',  # Default to first search tab
                            orientation='horizontal',
                            variant='pills',
                            color='blue',
                            radius='md',
                            styles={
                                'tab': {
                                    'fontSize': '14px',
                                    'fontWeight': 500,
                                    'padding': '10px 20px',
                                    'height': '42px',
                                    '&[data-active]': {
                                        'backgroundColor': '#1971c2',
                                        'color': 'white',
                                        'boxShadow': '0 2px 6px rgba(25, 113, 194, 0.4)'
                                    },
                                    '&:not([data-active])': {
                                        'backgroundColor': '#2c2e33',
                                        'color': '#909296'
                                    },
                                    '&:hover:not([data-active])': {
                                        'backgroundColor': '#373A40',
                                        'color': '#c1c2c5'
                                    }
                                }
                            },
                            className='mb-3'
                        ),
                        value='master-search'
                    ),

                    # Master Tab Panel 2: Data
                    dmc.TabsPanel(
                        dbc.Container([
                            dmc.Title("Data", order=2, mt='lg', mb='md'),
                            dmc.Text("Data information and downloads will go here.", size='md', c='dimmed')
                        ], fluid=True),
                        value='master-data'
                    ),

                    # Master Tab Panel 3: Paper
                    dmc.TabsPanel(
                        dbc.Container([
                            dmc.Title("Paper", order=2, mt='lg', mb='md'),
                            dmc.Text("Research paper and methodology will go here.", size='md', c='dimmed')
                        ], fluid=True),
                        value='master-paper'
                    ),

                    # Master Tab Panel 4: About
                    dmc.TabsPanel(
                        dbc.Container([
                            dmc.Title("About", order=2, mt='lg', mb='md'),
                            dmc.Text("About this project will go here.", size='md', c='dimmed')
                        ], fluid=True),
                        value='master-about'
                    ),

                    # Master Tab Panel 5: Help
                    dmc.TabsPanel(
                        dbc.Container([
                            dmc.Title("Help", order=2, mt='lg', mb='md'),
                            dmc.Text("Help documentation will go here.", size='md', c='dimmed')
                        ], fluid=True),
                        value='master-help'
                    ),
                ],
                id='master-tabs',
                value='master-search',  # Default to search master tab
                orientation='horizontal',
                variant='pills',
                color='cyan',
                radius='lg',
                styles={
                    'tab': {
                        'fontSize': '18px',
                        'fontWeight': 700,
                        'padding': '14px 28px',
                        'height': '52px',
                        '&[data-active]': {
                            'backgroundColor': '#0c8599',
                            'color': 'white',
                            'boxShadow': '0 4px 12px rgba(12, 133, 153, 0.6)'
                        },
                        '&:not([data-active])': {
                            'backgroundColor': '#25262b',
                            'color': '#868e96'
                        },
                        '&:hover:not([data-active])': {
                            'backgroundColor': '#2c2e33',
                            'color': '#adb5bd'
                        }
                    }
                },
                className='mb-4'
            ),

            # Map with floating panels (visible in search tab only)
            html.Div(
                id='search-map-container',
                children=[
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
                ]
            ),

            # Data stores
            dcc.Store(id='active-search-tab', data='state-similarity'),  # Track which tab is active
            dcc.Store(id='selected-location-id'),  # Shared across tabs

            # Store per search type (new naming pattern)
            dcc.Store(id='state-similarity-result-locations'),
            dcc.Store(id='state-similarity-query-params'),

            dcc.Store(id='dissimilarity-result-locations'),
            dcc.Store(id='dissimilarity-query-params'),

            dcc.Store(id='state-description-result-locations'),
            dcc.Store(id='state-description-query-params'),

            dcc.Store(id='change-similarity-result-locations'),
            dcc.Store(id='change-similarity-query-params'),

            dcc.Store(id='change-description-result-locations'),
            dcc.Store(id='change-description-query-params'),
        ]

        print(f"DEBUG: Layout created with {len(components)} components")
        # Wrap in MantineProvider with dark theme for dmc components
        return dmc.MantineProvider(
            dbc.Container(
                components,
                fluid=True
            ),
            theme={"colorScheme": "dark"}
        )


    # ------- Search handlers  ------- #
    def image_state_search(self, n_clicks, location_id, boroughs, year, target_year, limit, media_type, use_faiss, use_whitening, active_tab):
        from .. import context as app_ctx
        """Handle state search (image-to-image by year)."""
        # Only process if state tab is active
        #if active_tab != 'image-state':
        #    return dbc.Alert("Please switch to State Search tab", color='warning'), {'display': 'block'}, [], None

        if not location_id or not year:
            return (
                dbc.Alert("Please select a location and year", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        try:
            results_set = self.state_similarity_form.execute_search(
                app_ctx=app_ctx,
                location_id=location_id,
                boroughs=boroughs,
                year=year,
                target_year=target_year,
                limit=limit,
                media_type=media_type,
                use_faiss=use_faiss,
                use_whitening=use_whitening,
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
        
    def image_change_search(self, n_clicks, location_id, boroughs, year_from, year_to, limit, media_type, sequential_value, active_tab, use_faiss, use_whitening):
        """Handle change search (temporal change detection)."""

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
            results_set = self.change_similarity_form.execute_search(
                app_ctx=app_ctx,
                location_id=location_id,
                boroughs=boroughs,
                year_from=year_from,
                year_to=year_to,
                limit=limit,
                media_type=media_type,
                sequential=sequential,
                use_faiss=use_faiss,
                use_whitening=use_whitening
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
    
    def text_state_search(self, n_clicks, text, boroughs, target_year, limit, media_type, active_tab):
        from .. import context as app_ctx
        """Handle text state search (text-to-image by year)."""
        if not text:
            return (
                dbc.Alert("Please enter search text", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        try:
            results_set = self.state_description_form.execute_search(
                state=app_ctx,
                text=text,
                boroughs=boroughs,
                target_year=target_year,
                limit=limit,
                media_type=media_type
            )

            if len(results_set) == 0:
                return (
                    dbc.Alert(f"No results found", color='info'),
                    {'display': 'block'},
                    [],
                    {'target_year': target_year}
                )

            # Create results panel from results set
            results_panel = ResultsPanel(id_prefix='results', results=results_set)
            result_location_ids = [r.location_id for r in results_set]

            return (
                results_panel.content,
                {'display': 'block'},
                result_location_ids,
                {'target_year': target_year, 'text': text}
            )

        except Exception as e:
            logger.error(f"Text state search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                [],
                None
            )

    def text_change_search(self, n_clicks, text, boroughs, limit, media_type, sequential_value, active_tab):
        """Handle text change search (text-to-image temporal change detection)."""
        from .. import context as app_ctx

        if not text:
            return (
                dbc.Alert("Please enter search text", color='warning'),
                {'display': 'block'},
                [],
                None
            )

        # Convert sequential checkbox value to boolean
        sequential = 'sequential' in (sequential_value or [])

        try:
            results_set = self.change_description_form.execute_search(
                app_ctx=app_ctx,
                text=text,
                boroughs=boroughs,
                limit=limit,
                media_type=media_type,
                sequential=sequential
            )

            if len(results_set) == 0:
                return (
                    dbc.Alert(f"No results found", color='info'),
                    {'display': 'block'},
                    [],
                    {}
                )

            # Create results panel from results set
            results_panel = ResultsPanel(id_prefix='results', results=results_set)
            result_location_ids = [r.location_id for r in results_set]

            return (
                results_panel.content,
                {'display': 'block'},
                result_location_ids,
                {'sequential': sequential, 'text': text}
            )

        except Exception as e:
            logger.error(f"Text change search error: {e}", exc_info=True)
            return (
                dbc.Alert(f"Error: {str(e)}", color='danger'),
                {'display': 'block'},
                [],
                None
            )
        