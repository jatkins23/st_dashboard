"""Text-to-image search tab."""

from dash import dcc, html
from ...config import COLORS


def create_text_search_tab(available_years):
    """Create text-to-image search interface (horizontal layout).

    Args:
        available_years: List of available years

    Returns:
        Dash HTML Div with tab content
    """
    return html.Div([
        html.Div([
            # Text query input
            html.Div([
                html.Label("Text Query:"),
                dcc.Input(
                    id='text-query-input',
                    type='text',
                    placeholder='e.g., "street with trees and parked cars"',
                    style={'width': '100%'}
                ),
            ], className='search-form-field flex-1', style={'minWidth': '300px'}),

            # Year dropdown
            html.Div([
                html.Label("Year (optional):"),
                dcc.Dropdown(
                    id='text-year-dropdown',
                    options=[{'label': 'All Years', 'value': None}] +
                            [{'label': str(y), 'value': y} for y in available_years],
                    placeholder='All years',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '140px'}),

            # Limit dropdown
            html.Div([
                html.Label("Results:"),
                dcc.Dropdown(
                    id='text-limit-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                    value=10,
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '100px'}),

            # FAISS checkbox
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),
                dcc.Checklist(
                    id='text-use-faiss-checkbox',
                    options=[{'label': ' FAISS', 'value': 'faiss'}],
                    value=[],
                ),
            ], className='search-form-field', style={'minWidth': '100px'}),

            # Whitening checkbox
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),
                dcc.Checklist(
                    id='text-use-whitening-checkbox',
                    options=[{'label': ' Whitening', 'value': 'whitening'}],
                    value=[],
                ),
            ], className='search-form-field', style={'minWidth': '120px'}),

            # Search button
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),
                html.Button('Search', id='text-search-btn', n_clicks=0,
                           style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%'}),
            ], className='search-form-field', style={'minWidth': '100px'}),

        ], className='search-form-row'),

        # Results div
        html.Div(id='text-results', style={'marginTop': 30})
    ])
