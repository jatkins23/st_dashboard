"""Location similarity search tab."""

from dash import dcc, html
from ...config import COLORS
from dash.development.base_component import Component

def search_bar(available_years, type:str='state') -> Component:

    components = []
    state_search = html.Div([
        html.Label("Location ID:"),
        dcc.Input(
            id='location-id-input',
            type='number',
            placeholder='Enter location ID',
            style={'width': '150px'}
        ),
    ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'})

    # def year_picker(id: str, label: str | None = None, none_option: bool=False, placeholder=)
    year_picker = html.Div([
        html.Label("Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(y), 'value': y} for y in available_years],
            placeholder='Select year',
            style={'width': '150px'}
        ),
    ], style={'display': 'inline-blcok', 'marginRight': 20, 'verticalAlign': 'top'})

    target_year_picker = html.Div([
        html.Label("Target Year (optional):"),
        dcc.Dropdown(
            id='target-year-dropdown',
            options=[{'label': 'All Years', 'value': None}] +
                    [{'label': str(y), 'value': y} for y in available_years],
            placeholder='All years',
            style={'width': '150px'}
        ),
    ], style={'display': 'inline-block', 'marginRight': 20, 'verticalAlign': 'top'})

    limit_picker = html.Div([
        html.Label("Results:"),
        dcc.Dropdown(
            id='limit-dropdown',
            options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
            value=10,
            style={'width': '100px'}
        ),
    ], style={'display': 'inline-block', 'verticalAlign': 'top'})

    components = [state_search, year_picker, target_year_picker, limit_picker]

    return html.Div(components, style={'marginBottom': 20})


def create_state_search_tab(available_years):
    """Create location similarity search interface (horizontal layout).

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
                    id='location-id-input',
                    type='number',
                    placeholder='Enter location ID',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '140px'}),

            # Year dropdown
            html.Div([
                html.Label("Year:"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(y), 'value': y} for y in available_years],
                    placeholder='Select year',
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '120px'}),

            # Target year dropdown
            html.Div([
                html.Label("Target Year (optional):"),
                dcc.Dropdown(
                    id='target-year-dropdown',
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
                    id='limit-dropdown',
                    options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                    value=10,
                    style={'width': '100%'}
                ),
            ], className='search-form-field', style={'minWidth': '100px'}),

            # FAISS checkbox
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),  # Spacer for alignment
                dcc.Checklist(
                    id='use-faiss-checkbox',
                    options=[{'label': ' FAISS', 'value': 'faiss'}],
                    value=[],
                ),
            ], className='search-form-field', style={'minWidth': '100px'}),

            # Whitening checkbox
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),  # Spacer for alignment
                dcc.Checklist(
                    id='use-whitening-checkbox',
                    options=[{'label': ' Whitening', 'value': 'whitening'}],
                    value=[],
                ),
            ], className='search-form-field', style={'minWidth': '120px'}),

            # Search button
            html.Div([
                html.Label("\u00A0", style={'visibility': 'hidden'}),  # Spacer for alignment
                html.Button('Search', id='location-search-btn', n_clicks=0,
                           style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%'}),
            ], className='search-form-field', style={'minWidth': '100px'}),

        ], className='search-form-row'),

        # Results div
        html.Div(id='location-results', style={'marginTop': 30})
    ])
