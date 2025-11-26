"""Location similarity search tab."""

from dash import dcc, html
from ...config import COLORS


def create_location_search_tab(available_years):
    """Create location similarity search interface.

    Args:
        available_years: List of available years

    Returns:
        Dash HTML Div with tab content
    """
    return html.Div([
        html.H3("Find Similar Locations"),

        html.Div([
            html.Div([
                html.Div([
                    html.Label("Location ID:"),
                    dcc.Input(
                        id='location-id-input',
                        type='number',
                        placeholder='Enter location ID',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Year:"),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': str(y), 'value': y} for y in available_years],
                        placeholder='Select year',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Target Year (optional):"),
                    dcc.Dropdown(
                        id='target-year-dropdown',
                        options=[{'label': 'All Years', 'value': None}] +
                                [{'label': str(y), 'value': y} for y in available_years],
                        placeholder='All years',
                        style={'width': '150px'}
                    ),
                ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'}),

                html.Div([
                    html.Label("Results:"),
                    dcc.Dropdown(
                        id='limit-dropdown',
                        options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                        value=10,
                        style={'width': '100px'}
                    ),
                ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'marginBottom': 20}),
        ]),

        html.Div([
            html.Label("Search Options:", style={'fontWeight': 'bold', 'marginBottom': 10}),
            html.Div([
                dcc.Checklist(
                    id='use-faiss-checkbox',
                    options=[{'label': ' Use FAISS (faster search)', 'value': 'faiss'}],
                    value=[],
                    style={'marginBottom': 5}
                ),
                dcc.Checklist(
                    id='use-whitening-checkbox',
                    options=[{'label': ' Use whitening reranking (better quality)', 'value': 'whitening'}],
                    value=[],
                    style={'marginBottom': 5}
                ),
            ], style={'marginBottom': 15}),
        ]),

        html.Button('Search', id='location-search-btn', n_clicks=0,
                   style={'padding': '10px 24px', 'fontSize': 16}),

        html.Div(id='location-results', style={'marginTop': 30})
    ])
