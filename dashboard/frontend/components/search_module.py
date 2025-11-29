"""Modular search bar component."""

from dash import dcc, html
from ...config import COLORS


def create_search_bar(available_years: list):
    """Create the main search bar with tabs and search fields.

    Args:
        available_years: List of available years for dropdowns

    Returns:
        Dash HTML Div with search bar
    """
    return html.Div([
        # Tabs
        dcc.Tabs(
            id='search-tabs',
            value='state',
            children=[
                dcc.Tab(label='State', value='state', className='tab'),
                dcc.Tab(label='Text', value='text', className='tab'),
                dcc.Tab(label='Change', value='change', className='tab'),
                dcc.Tab(label='Stats', value='stats', className='tab'),
            ],
            style={'marginBottom': '15px'}
        ),

        # Search fields (populated by callback based on tab)
        html.Div(id='search-fields')
    ])


def get_state_search_fields(available_years: list):
    """Get search fields for State/Location tab."""
    return html.Div([
        html.Div([
            html.Label("Location ID:"),
            dcc.Input(
                id='location-id-input',
                type='number',
                placeholder='Enter location ID',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '140px'}),

        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        html.Div([
            html.Label("Target Year:"),
            dcc.Dropdown(
                id='target-year-dropdown',
                options=[{'label': 'All', 'value': None}] +
                        [{'label': str(y), 'value': y} for y in available_years],
                placeholder='All years',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '130px'}),

        html.Div([
            html.Label("Results:"),
            dcc.Dropdown(
                id='limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        html.Div([
            dcc.Checklist(
                id='use-faiss-checkbox',
                options=[{'label': ' FAISS', 'value': 'faiss'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field'),

        html.Div([
            dcc.Checklist(
                id='use-whitening-checkbox',
                options=[{'label': ' Whitening', 'value': 'whitening'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field'),

        html.Div([
            html.Button('Search', id='search-btn', n_clicks=0,
                       style={'marginTop': '20px', 'padding': '10px 24px'}),
        ], className='search-form-field'),

    ], className='search-form-row')


def get_text_search_fields(available_years: list):
    """Get search fields for Text tab."""
    return html.Div([
        html.Div([
            html.Label("Text Query:"),
            dcc.Input(
                id='text-query-input',
                type='text',
                placeholder='e.g., "street with trees and parked cars"',
                style={'width': '100%'}
            ),
        ], className='search-form-field flex-1', style={'minWidth': '300px'}),

        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id='text-year-dropdown',
                options=[{'label': 'All', 'value': None}] +
                        [{'label': str(y), 'value': y} for y in available_years],
                placeholder='All years',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '130px'}),

        html.Div([
            html.Label("Results:"),
            dcc.Dropdown(
                id='text-limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        html.Div([
            dcc.Checklist(
                id='text-use-faiss-checkbox',
                options=[{'label': ' FAISS', 'value': 'faiss'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field'),

        html.Div([
            dcc.Checklist(
                id='text-use-whitening-checkbox',
                options=[{'label': ' Whitening', 'value': 'whitening'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field'),

        html.Div([
            html.Button('Search', id='text-search-btn', n_clicks=0,
                       style={'marginTop': '20px', 'padding': '10px 24px'}),
        ], className='search-form-field'),

    ], className='search-form-row')


def get_change_search_fields(available_years: list):
    """Get search fields for Change tab."""
    return html.Div([
        html.Div([
            html.Label("Location ID:"),
            dcc.Input(
                id='change-location-id-input',
                type='number',
                placeholder='Enter location ID',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '140px'}),

        html.Div([
            html.Label("From Year:"),
            dcc.Dropdown(
                id='year-from-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        html.Div([
            html.Label("To Year:"),
            dcc.Dropdown(
                id='year-to-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        html.Div([
            html.Label("Results:"),
            dcc.Dropdown(
                id='change-limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        html.Div([
            html.Button('Search', id='change-search-btn', n_clicks=0,
                       style={'marginTop': '20px', 'padding': '10px 24px'}),
        ], className='search-form-field'),

    ], className='search-form-row')


def get_stats_fields():
    """Get fields for Stats tab."""
    return html.Div([
        html.Button('Refresh Stats', id='stats-refresh-btn', n_clicks=0,
                   style={'padding': '10px 24px'}),
    ])
