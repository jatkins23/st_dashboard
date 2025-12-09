"""Search form component for image-to-image state search."""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent
from dash import Input, Output, State

from .base import BaseComponent
from .. import state

from streettransformer.db.database import get_connection

import logging
logger = logging.getLogger(__name__)

class SearchForm(BaseComponent):
    """Search form component for ImageToImage state search.

    This component provides a form with:
    - Street selector (MultiSelect)
    - Year dropdown
    - Target year dropdown (optional)
    - Limit dropdown
    - Media type dropdown
    - Options checkboxes (FAISS, Whitening)
    - Search button
    """

    def __init__(self, available_years: list, all_streets: list = None, id_prefix: str = 'search-form', title: str = "Image-to-Image State Search"):
        """Initialize the search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            id_prefix: Prefix for component IDs
        """
        super().__init__(id_prefix=id_prefix)
        self.title = title
        self.available_years = available_years
        self.all_streets = all_streets or []

    def register_callbacks(self, app):
        """Register callbacks for the search form.

        Registers the street filtering callback that updates available street options
        based on selected streets to show only valid combinations.
        """
        logger = logging.getLogger(__name__)

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
    
    def _street_selector(self, id="street-selector") -> DashComponent:
        multi_select = dmc.MultiSelect(
            id=id,
            label="Select Streets",
            placeholder="Start typing streets...",
            data = [{"label": s, "value": s} for s in self.all_streets],
            searchable=True,
            nothingFoundMessage="No options",
            clearable=True,
            maxValues=4,
            hidePickedOptions=True,
            maxDropdownHeight=300,
            limit=10,
            value=[],
            styles={
                "input": {"color": "black"},
                "dropdown": {"color": "black", "zIndex": 9999}
            }
        )
        logger.info(f"DMC MultiSelect created successfully with {len(self.all_streets)} streets")
        return multi_select
        
    def _year_selector(self, id='year-dropdown', include_all:bool=False) -> DashComponent:
        options = [{'label': str(y), 'value': y} for y in self.available_years]
        if include_all:
            options = [{'label': 'All', 'value': None}] + options
            
        return dcc.Dropdown(
            id=id,
            options=options,
            placeholder='Select year',
            style={"dropdown": {"color": "black", "zIndex": 9999}}
        )
        
    def _media_selector(self, id='media-type-selector') -> DashComponent:
        return dcc.Dropdown(
            id=id,
            # TODO: make more modular
            options=[
                {'label': 'Images', 'value': 'image'},
                {'label': 'Masks', 'value': 'mask'},
                {'label': 'Side-by-side', 'value': 'sidebyside'}
            ],
            value='image',
            clearable=False,
            style={"dropdown": {"color": "black", "zIndex": 9999}}
            
        )
        
    def _method_selector(self, options:list=['faiss', 'whitening']) -> DashComponent:
        if not options or len(options) == 0:
            return [] # TODO: maybe an alert here?
        
        components = []
        components.append(dbc.Label("Options", size='sm'))
        
        if 'faiss' in options:
            components.append(
                dbc.Checklist(
                    id='use-faiss-checkbox',
                    options=[{'label': ' FAISS', 'value': 'faiss'}],
                    value=['faiss'],
                    switch=True
                )
            )
        if 'whitening' in options:
            components.append(
                dbc.Checklist(
                    id='use-whitening-checkbox',
                    options=[{'label': ' Whitening', 'value': 'whitening'}],
                    value=[],
                    switch=True
                )
            )
        
        return components
        
    
    @property
    def layout(self) -> DashComponent:
        """Return the search form layout."""
        search_components = [
            
            # Street Selector
            dbc.Col([self._street_selector()], width=3),
            
            # Year Dropdown
            dbc.Col([
                dbc.Label("Year", size='sm'),
                self._year_selector(id='year-selector')
            ], width=2),

            # Target Year Dropdown
            dbc.Col([
                dbc.Label("Target Year (optional)", size='sm'),
                self._year_selector(id='target-year-selector')
            ], width=2),

            # Limit Dropdown
            dbc.Col([
                dbc.Label("Limit", size='sm'),
                dcc.Dropdown(
                    id='limit-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                    value=10
                )
            ], width=1),

            # Media Type Dropdown
            dbc.Col([
                dbc.Label("Media Type", size='sm'),
                self._media_selector()
            ], width=2),

            # Options Checkboxes
            dbc.Col([self._method_selector(options=['faiss','whitening'])], width=2),

            # Search Button
            dbc.Col([
                dbc.Button(
                    'Search',
                    id='search-btn',
                    color='primary',
                    className='mt-4'
                )
            ], width=1),

        ]
        card_layout = dbc.Card([
            dbc.CardHeader("Image-to-Image State Search", className='fw-bold'),
            dbc.CardBody([
                dbc.Row(
                    search_components,
                    className='g-3', align='end')
            ])
        ])
        return card_layout
