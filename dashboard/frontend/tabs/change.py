"""Change pattern detection tab."""

from dash import dcc, html
from ...config import COLORS


def create_change_search_tab(available_years):
    """Create change pattern search interface (horizontal layout).

    Args:
        available_years: List of available years

    Returns:
        Dash HTML Div with tab content
    """
    return html.Div([
        html.Div([
            # Location ID input
            html.Div([
                html.Label("Location ID:"),
                dcc.Input(
                    id='change-location-id-input',
                    type='number',
                    placeholder='Enter location ID',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '140px'}),

            # From Year dropdown
            html.Div([
                html.Label("From Year:"),
                dcc.Dropdown(
                    id='year-from-dropdown',
                    options=[{'label': str(y), 'value': y} for y in available_years],
                    placeholder='Select year',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '120px'}),

            # To Year dropdown
            html.Div([
                html.Label("To Year:"),
                dcc.Dropdown(
                    id='year-to-dropdown',
                    options=[{'label': str(y), 'value': y} for y in available_years],
                    placeholder='Select year',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '120px'}),

            # Limit dropdown
            html.Div([
                html.Label("Results:"),
                dcc.Dropdown(
                    id='change-limit-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                    value=10,
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '100px'}),

            # Search button
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),
                html.Button('Search', id='change-search-btn', n_clicks=0,
                           style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%'}),
            ], className='search-form-field', style={'minWidth': '100px'}),

        ], className='search-form-row'),

        # Results div
        html.Div(id='change-results', style={'marginTop': 30})
    ])
