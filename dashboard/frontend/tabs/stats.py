"""Statistics tab."""

from dash import html


def create_stats_tab():
    """Create statistics overview interface.

    Returns:
        Dash HTML Div with tab content
    """
    return html.Div([
        html.H3("Embedding Statistics"),
        html.Button('Refresh Stats', id='stats-refresh-btn', n_clicks=0,
                   style={'padding': '10px 24px', 'fontSize': 16, 'marginBottom': 20}),
        html.Div(id='stats-content')
    ])
