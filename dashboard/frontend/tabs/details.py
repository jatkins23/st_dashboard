"""Details tab for selected location."""

from dash import dcc, html
from ...config import COLORS


def create_details_tab():
    """Create location details display interface.

    Returns:
        Dash HTML Div with details content
    """
    return html.Div([
        html.Div(id='location-details-content', children=[
            html.Div([
                html.P("Click on a location on the map to view details",
                      style={
                          'color': COLORS['text-secondary'],
                          'textAlign': 'center',
                          'padding': '40px 20px',
                          'fontStyle': 'italic'
                      })
            ])
        ])
    ])
