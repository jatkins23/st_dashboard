"""Unified search bar component that adapts to different tabs."""

from dash import dcc, html
from ...config import COLORS


def create_search_fields(tab: str, available_years: list):
    """Create search fields based on active tab.

    Args:
        tab: Active tab identifier ('state', 'text', 'change', 'stats')
        available_years: List of available years for dropdowns

    Returns:
        Dash HTML Div with appropriate search fields for the tab
    """
    if tab == 'state':
        return _create_state_search_fields(available_years)
    elif tab == 'text':
        return _create_text_search_fields(available_years)
    elif tab == 'change':
        return _create_change_search_fields(available_years)
    elif tab == 'stats':
        return _create_stats_search_fields()
    else:
        return html.Div()


def _create_state_search_fields(available_years: list):
    """Create search fields for state/location similarity search."""
    return html.Div([
        # Location ID input
        html.Div([
            html.Label("Location ID:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Input(
                id='location-id-input',
                type='number',
                placeholder='Enter location ID',
                style={'width': '100%', 'padding': '8px'}
            ),
        ], className='search-form-field', style={'minWidth': '140px'}),

        # Year dropdown
        html.Div([
            html.Label("Year:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        # Target year dropdown
        html.Div([
            html.Label("Target Year (optional):", style={'fontSize': '12px', 'marginBottom': '4px'}),
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
            html.Label("Results:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        # FAISS checkbox
        html.Div([
            dcc.Checklist(
                id='use-faiss-checkbox',
                options=[{'label': ' FAISS', 'value': 'faiss'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        # Whitening checkbox
        html.Div([
            dcc.Checklist(
                id='use-whitening-checkbox',
                options=[{'label': ' Whitening', 'value': 'whitening'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        # Search button
        html.Div([
            html.Button('Search', id='location-search-btn', n_clicks=0,
                       style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%', 'marginTop': '20px'}),
        ], className='search-form-field', style={'minWidth': '100px'}),

    ], className='search-form-row')


def _create_text_search_fields(available_years: list):
    """Create search fields for text-to-image search."""
    return html.Div([
        # Text query input
        html.Div([
            html.Label("Text Query:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Input(
                id='text-query-input',
                type='text',
                placeholder='e.g., "street with trees and parked cars"',
                style={'width': '100%', 'padding': '8px'}
            ),
        ], className='search-form-field flex-1', style={'minWidth': '300px'}),

        # Year dropdown
        html.Div([
            html.Label("Year (optional):", style={'fontSize': '12px', 'marginBottom': '4px'}),
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
            html.Label("Results:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='text-limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        # FAISS checkbox
        html.Div([
            dcc.Checklist(
                id='text-use-faiss-checkbox',
                options=[{'label': ' FAISS', 'value': 'faiss'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        # Whitening checkbox
        html.Div([
            dcc.Checklist(
                id='text-use-whitening-checkbox',
                options=[{'label': ' Whitening', 'value': 'whitening'}],
                value=[],
                style={'marginTop': '20px'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        # Search button
        html.Div([
            html.Button('Search', id='text-search-btn', n_clicks=0,
                       style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%', 'marginTop': '20px'}),
        ], className='search-form-field', style={'minWidth': '100px'}),

    ], className='search-form-row')


def _create_change_search_fields(available_years: list):
    """Create search fields for change pattern detection."""
    return html.Div([
        # Location ID input
        html.Div([
            html.Label("Location ID:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Input(
                id='change-location-id-input',
                type='number',
                placeholder='Enter location ID',
                style={'width': '100%', 'padding': '8px'}
            ),
        ], className='search-form-field', style={'minWidth': '140px'}),

        # From Year dropdown
        html.Div([
            html.Label("From Year:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='year-from-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        # To Year dropdown
        html.Div([
            html.Label("To Year:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='year-to-dropdown',
                options=[{'label': str(y), 'value': y} for y in available_years],
                placeholder='Select year',
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '120px'}),

        # Limit dropdown
        html.Div([
            html.Label("Results:", style={'fontSize': '12px', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='change-limit-dropdown',
                options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
                value=10,
                style={'width': '100%'}
            ),
        ], className='search-form-field', style={'minWidth': '100px'}),

        # Search button
        html.Div([
            html.Button('Search', id='change-search-btn', n_clicks=0,
                       style={'padding': '10px 24px', 'fontSize': 14, 'width': '100%', 'marginTop': '20px'}),
        ], className='search-form-field', style={'minWidth': '100px'}),

    ], className='search-form-row')


def _create_stats_search_fields():
    """Create search fields for statistics tab."""
    return html.Div([
        html.Button('Refresh Stats', id='stats-refresh-btn', n_clicks=0,
                   style={'padding': '10px 24px', 'fontSize': 14}),
    ], style={'display': 'flex', 'gap': '10px'})
