"""Clean, map-centric dashboard layout."""

from dash import dcc, html
import dash_bootstrap_components as dbc
from ..config import COLORS


def create_app_with_styling():
    """Create Dash app with external CSS.

    Returns:
        Tuple of (app, css_path)
    """
    import dash
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder='frontend/static'
    )

    # Simple index with accordion script
    index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
            <script>
                // Simple accordion toggle
                document.addEventListener('click', function(e) {
                    if (e.target && e.target.classList.contains('accordion-header')) {
                        const content = e.target.nextElementSibling;
                        content.classList.toggle('active');
                    }
                });
            </script>
        </body>
    </html>
    '''

    return app, index_string


def layout(universe_name: str):
    """Create clean map-centric layout.

    Args:
        universe_name: Name of the universe being explored

    Returns:
        Dash HTML Div with complete layout
    """
    return html.Div([
        # Search bar at the top
        dbc.Row(
            id='search-bar-container',
            style={
                'backgroundColor': COLORS['card'],
                'padding': '12px 20px',
                'borderBottom': f'1px solid {COLORS["border"]}',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.3)',
                'zIndex': 1000,
                'position': 'relative',
                'display': 'flex'
            }
        ),

        # Map and panels container
        dbc.Row([
            # Left panel - Results
            dbc.Col([
                html.Div([
                    html.Div([
                        html.H3("Results", style={
                            'margin': '0 0 15px 0',
                            'fontSize': '18px',
                            'color': COLORS['text']
                        }),
                        html.Div(id='results-content', style={
                            'overflowY': 'auto',
                            'maxHeight': 'calc(100vh - 200px)'
                        })
                    ], style={'padding': '20px'})
                ], 
                id='results-panel', style={
                    'position': 'absolute',
                    'left': '20px',
                    'top': '20px',
                    'bottom': '20px',
                    'width': '380px',
                    'backgroundColor': COLORS['card'],
                    'borderRadius': '8px',
                    'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
                    'zIndex': 100,
                    'display': 'none'  # Hidden by default
                }),
            ]),

            # Map background
            html.Div([
                dcc.Graph(
                    id='main-map',
                    style={'width': '100%', 'height': '100%'},
                    config={'displayModeBar': False, 'scrollZoom': True}
                ),
            ], style={
                'position': 'absolute',
                'top': 0,
                'left': 0,
                'right': 0,
                'bottom': 0,
                'zIndex': 1
            }),

            # Right panel - Details
            html.Div([
                html.Div([
                    html.H3("Location Details", style={
                        'margin': '0 0 15px 0',
                        'fontSize': '18px',
                        'color': COLORS['text']
                    }),
                    html.Div(id='details-content', style={
                        'overflowY': 'auto',
                        'maxHeight': 'calc(100vh - 200px)'
                    })
                ], style={'padding': '20px'})
            ], id='details-panel', style={
                'position': 'absolute',
                'right': '20px',
                'top': '20px',
                'bottom': '20px',
                'width': '420px',
                'backgroundColor': COLORS['card'],
                'borderRadius': '8px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
                'zIndex': 100,
                'display': 'none'  # Hidden by default
            }),

        ], style={
            'position': 'relative',
            'height': 'calc(100vh - 100px)',  # Subtract search bar height (now more compact)
            'overflow': 'hidden'
        }),

        # Data stores
        dcc.Store(id='query-location-id'),
        dcc.Store(id='result-locations'),
        dcc.Store(id='all-locations'),

    ], style={'width': '100vw', 'height': '100vh', 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column'})
