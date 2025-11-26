"""Change pattern detection tab."""

from dash import dcc, html
from ...config import COLORS


def create_change_search_tab(available_years):
    """Create change pattern search interface.

    Args:
        available_years: List of available years

    Returns:
        Dash HTML Div with tab content
    """
    return html.Div([
        html.H3("Change Pattern Detection"),
        html.P("Find locations with similar changes between years",
               style={'color': COLORS['text-secondary']}),

        html.Div([
            html.Div([
                html.Div([
                    html.Label("Location ID:"),
                    dcc.Input(
                        id='change-location-id-input',
                        type='number',
                        placeholder='Enter location ID',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("From Year:"),
                    dcc.Dropdown(
                        id='year-from-dropdown',
                        options=[{'label': str(y), 'value': y} for y in available_years],
                        placeholder='Select year',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("To Year:"),
                    dcc.Dropdown(
                        id='year-to-dropdown',
                        options=[{'label': str(y), 'value': y} for y in available_years],
                        placeholder='Select year',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Results:"),
                    dcc.Dropdown(
                        id='change-limit-dropdown',
                        options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                        value=10,
                        style={'width': '100px'}
                    ),
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'marginBottom': 20}),
        ]),

        html.Button('Search', id='change-search-btn', n_clicks=0,
                   style={'padding': '10px 24px', 'fontSize': 16}),

        html.Div(id='change-results', style={'marginTop': 30})
    ])
