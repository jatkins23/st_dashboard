"""Simple Bootstrap-based dashboard layout."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from .components.results.results_panel import ResultsPanel
from .components.details.details_panel import DetailsPanel


def create_app():
    """Create Dash app with Bootstrap theme."""
    import dash
    from pathlib import Path

    # Get path to assets folder
    assets_folder = Path(__file__).parent.parent / 'assets'

    return dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=str(assets_folder)
    )

def create_layout(universe_name: str):
    """Create simple layout with state search.

    Args:
        universe_name: Name of the universe being explored

    Returns:
        Bootstrap Container with layout
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H3(f"Street Transformer Dashboard - {universe_name}", className='text-light mb-3')
            ])
        ], className='mt-3'),

        # Search card
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Image-to-Image State Search", className='fw-bold'),
                    dbc.CardBody([
                        html.Div(id='search-form')
                    ])
                ])
            ])
        ], className='mb-3'),

        # Map with floating panels
        dbc.Row([
            # Map container with floating results panel
            dbc.Col([
                html.Div([
                    # Map
                    dcc.Graph(
                        id='main-map',
                        style={'height': '60vh'},
                        config={'displayModeBar': False, 'scrollZoom': True}
                    ),
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

    ], fluid=True)


def create_search_form(available_years: list):
    """Create search form for ImageToImage state search.

    Args:
        available_years: List of available years

    Returns:
        Bootstrap form row with search fields
    """
    return dbc.Row([
        dbc.Col([
            dbc.Label("Location ID", size='sm'),
            dbc.Input(
                id='location-id-input',
                type='number',
                placeholder='Enter location ID',
                size='sm'
            )
        ], width=2),

        dbc.Col([
            dbc.Label("Year", size='sm'),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year'
            )
        ], width=2),

        dbc.Col([
            dbc.Label("Target Year (optional)", size='sm'),
            dcc.Dropdown(
                id='target-year-dropdown',
                options=[{'label': 'All', 'value': None}] +
                        [{'label': str(y), 'value': y} for y in available_years],
                placeholder='All years'
            )
        ], width=2),

        dbc.Col([
            dbc.Label("Limit", size='sm'),
            dcc.Dropdown(
                id='limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10
            )
        ], width=1),

        dbc.Col([
            dbc.Label("Options", size='sm'),
            dbc.Checklist(
                id='use-faiss-checkbox',
                options=[{'label': ' FAISS', 'value': 'faiss'}],
                value=['faiss'],
                switch=True
            ),
            dbc.Checklist(
                id='use-whitening-checkbox',
                options=[{'label': ' Whitening', 'value': 'whitening'}],
                value=[],
                switch=True
            )
        ], width=2),

        dbc.Col([
            dbc.Button(
                'Search',
                id='search-btn',
                color='primary',
                className='mt-4'
            )
        ], width=1),

    ], className='g-3', align='end')