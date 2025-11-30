"""Main dashboard layout and app creation."""

from dash import dcc, html
from ..config import COLORS


def create_app_with_styling():
    """Create Dash app with dark mode styling.

    Returns:
        Tuple of (app, index_string) where index_string contains CSS styling
    """
    import dash
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    # Dark mode styling
    index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: ''' + COLORS['background'] + ''';
                    color: ''' + COLORS['text'] + ''';
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                .accordion-item {
                    background-color: ''' + COLORS['card'] + ''';
                    border: 1px solid ''' + COLORS['border'] + ''';
                    border-radius: 6px;
                    margin-bottom: 10px;
                    overflow: hidden;
                }
                .accordion-header {
                    padding: 15px 20px;
                    cursor: pointer;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background-color: ''' + COLORS['card'] + ''';
                    transition: background-color 0.2s;
                }
                .accordion-header:hover {
                    background-color: ''' + COLORS['input-bg'] + ''';
                }
                .accordion-content {
                    padding: 20px;
                    border-top: 1px solid ''' + COLORS['border'] + ''';
                    display: none;
                }
                .accordion-content.active {
                    display: block;
                }
                .similarity-badge {
                    background-color: ''' + COLORS['primary'] + ''';
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 14px;
                    font-weight: 500;
                }
                .location-info {
                    color: ''' + COLORS['text-secondary'] + ''';
                    font-size: 14px;
                    margin-top: 5px;
                }
                .image-container {
                    margin-top: 15px;
                    text-align: center;
                }
                .image-container img {
                    max-width: 100%;
                    border-radius: 4px;
                    border: 1px solid ''' + COLORS['border'] + ''';
                }
                input, select, .Select-control {
                    background-color: ''' + COLORS['input-bg'] + ''' !important;
                    color: ''' + COLORS['text'] + ''' !important;
                    border: 1px solid ''' + COLORS['border'] + ''' !important;
                    border-radius: 4px !important;
                    padding: 8px 12px !important;
                }
                button {
                    background-color: ''' + COLORS['primary'] + ''';
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                button:hover {
                    background-color: ''' + COLORS['primary-hover'] + ''';
                }
                .tab {
                    background-color: ''' + COLORS['card'] + ''' !important;
                    color: ''' + COLORS['text'] + ''' !important;
                    border: 1px solid ''' + COLORS['border'] + ''' !important;
                }
                .tab--selected {
                    background-color: ''' + COLORS['primary'] + ''' !important;
                    color: white !important;
                }
                label {
                    color: ''' + COLORS['text-secondary'] + ''';
                    font-size: 14px;
                    font-weight: 500;
                    display: block;
                    margin-bottom: 6px;
                }
                .error-message {
                    color: ''' + COLORS['error'] + ''';
                    padding: 12px;
                    background-color: rgba(244, 135, 113, 0.1);
                    border-radius: 4px;
                    border-left: 3px solid ''' + COLORS['error'] + ''';
                }
                .warning-message {
                    color: ''' + COLORS['warning'] + ''';
                    padding: 12px;
                    background-color: rgba(206, 145, 120, 0.1);
                    border-radius: 4px;
                    border-left: 3px solid ''' + COLORS['warning'] + ''';
                }
                .success-message {
                    color: ''' + COLORS['success'] + ''';
                }
                h1, h2, h3, h4 {
                    color: ''' + COLORS['text'] + ''';
                }

                /* Search form horizontal layout */
                .search-form-row {
                    display: flex;
                    gap: 15px;
                    align-items: flex-end;
                    flex-wrap: wrap;
                }

                .search-form-field {
                    display: flex;
                    flex-direction: column;
                    min-width: 120px;
                }

                .search-form-field.flex-1 {
                    flex: 1;
                }

                .search-form-field label {
                    margin-bottom: 4px;
                    font-size: 13px;
                }
            </style>
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


def create_layout(universe_name: str):
    """Create main dashboard layout with map and floating panels.

    Args:
        universe_name: Name of the universe being explored

    Returns:
        Dash HTML Div with complete layout
    """
    return html.Div([
        # Search bar container at top
        html.Div([
            # Title and tabs row
            html.Div([
                html.H2(f"Image Embedding Explorer - {universe_name}",
                       style={'margin': 0, 'color': COLORS['text'], 'fontSize': '20px'}),
                dcc.Tabs(id='tabs', value='state', children=[
                    dcc.Tab(label='State', value='state', className='tab'),
                    dcc.Tab(label='Text', value='text', className='tab'),
                    dcc.Tab(label='Change', value='change', className='tab'),
                    dcc.Tab(label='Stats', value='stats', className='tab'),
                ], style={'marginLeft': '20px'}),
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '15px'
            }),

            # Search fields container (populated by callback based on tab)
            html.Div(id='search-fields-container', style={'marginBottom': '10px'}),

        ], style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'right': 0,
            'backgroundColor': COLORS['background'],
            'padding': '15px 20px',
            'borderBottom': f"1px solid {COLORS['border']}",
            'zIndex': 1000,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.3)'
        }),

        # Map container (full page below search bar)
        html.Div([
            dcc.Graph(
                id='location-map',
                style={'width': '100%', 'height': '100%'},
                config={'displayModeBar': False}
            ),

            # Floating Results Panel (left side)
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Results", style={'margin': 0, 'fontSize': '18px', 'color': COLORS['text']}),
                        html.Button('×', id='close-results-btn',
                                  style={
                                      'background': 'none',
                                      'border': 'none',
                                      'fontSize': '24px',
                                      'cursor': 'pointer',
                                      'color': COLORS['text-secondary'],
                                      'padding': '0',
                                      'width': '30px',
                                      'height': '30px'
                                  })
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}),
                    html.Div(id='results-panel-content', style={'overflowY': 'auto', 'maxHeight': 'calc(100% - 50px)'})
                ], style={'padding': '20px', 'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
            ], id='results-panel', style={
                'position': 'absolute',
                'left': '20px',
                'top': '20px',
                'bottom': '20px',
                'width': '400px',
                'backgroundColor': COLORS['card'],
                'borderRadius': '8px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
                'display': 'none',  # Hidden by default
                'zIndex': 100
            }),

            # Floating Detail Panel (right side)
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Location Details", style={'margin': 0, 'fontSize': '18px', 'color': COLORS['text']}),
                        html.Button('×', id='close-details-btn',
                                  style={
                                      'background': 'none',
                                      'border': 'none',
                                      'fontSize': '24px',
                                      'cursor': 'pointer',
                                      'color': COLORS['text-secondary'],
                                      'padding': '0',
                                      'width': '30px',
                                      'height': '30px'
                                  })
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}),
                    html.Div(id='details-panel-content', style={'overflowY': 'auto', 'maxHeight': 'calc(100% - 50px)'})
                ], style={'padding': '20px', 'height': '100%', 'display': 'flex', 'flexDirection': 'column'})
            ], id='details-panel', style={
                'position': 'absolute',
                'right': '20px',
                'top': '20px',
                'bottom': '20px',
                'width': '450px',
                'backgroundColor': COLORS['card'],
                'borderRadius': '8px',
                'boxShadow': '0 4px 12px rgba(0,0,0,0.5)',
                'display': 'none',  # Hidden by default
                'zIndex': 100
            }),

        ], style={
            'position': 'fixed',
            'top': '130px',  # Below search bar
            'left': 0,
            'right': 0,
            'bottom': 0,
            'backgroundColor': COLORS['background']
        }),

        # Hidden stores for state management
        dcc.Store(id='selected-location-id'),
        dcc.Store(id='result-location-ids'),
        dcc.Store(id='all-locations-data'),

    ], style={'height': '100vh', 'overflow': 'hidden'})
