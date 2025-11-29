"""Clean, map-centric dashboard layout."""

from dash import dcc, html
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


def create_clean_layout(universe_name: str):
    """Create clean map-centric layout.

    Args:
        universe_name: Name of the universe being explored

    Returns:
        Dash HTML Div with complete layout
    """
    return html.Div([
        # Full-page map background
        html.Div([
            dcc.Graph(
                id='main-map',
                style={'width': '100%', 'height': '100%'},
                config={'displayModeBar': False, 'scrollZoom': True}
            ),
        ], style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'right': 0,
            'bottom': 0,
            'zIndex': 1
        }),

        # Search bar (floating on top)
        html.Div(
            id='search-bar-container',
            style={
                'position': 'fixed',
                'top': '20px',
                'left': '50%',
                'transform': 'translateX(-50%)',
                'zIndex': 1000,
                'backgroundColor': COLORS['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
                'minWidth': '800px',
                'maxWidth': '90%'
            }
        ),

        # Left panel - Results
        html.Div([
            html.Div([
                html.H3("Results", style={
                    'margin': '0 0 15px 0',
                    'fontSize': '18px',
                    'color': COLORS['text']
                }),
                html.Div(id='results-content', style={
                    'overflowY': 'auto',
                    'maxHeight': 'calc(100vh - 180px)'
                })
            ], style={'padding': '20px'})
        ], id='results-panel', style={
            'position': 'fixed',
            'left': '20px',
            'top': '140px',
            'bottom': '20px',
            'width': '380px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '8px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
            'zIndex': 100,
            'display': 'none'  # Hidden by default
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
                    'maxHeight': 'calc(100vh - 180px)'
                })
            ], style={'padding': '20px'})
        ], id='details-panel', style={
            'position': 'fixed',
            'right': '20px',
            'top': '140px',
            'bottom': '20px',
            'width': '420px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '8px',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
            'zIndex': 100,
            'display': 'none'  # Hidden by default
        }),

        # Data stores
        dcc.Store(id='query-location-id'),
        dcc.Store(id='result-locations'),
        dcc.Store(id='all-locations'),

    ], style={'width': '100vw', 'height': '100vh', 'overflow': 'hidden'})
