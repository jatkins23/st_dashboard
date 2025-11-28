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
    """Create main dashboard layout with tabs.

    Args:
        universe_name: Name of the universe being explored

    Returns:
        Dash HTML Div with complete layout
    """
    return html.Div([
        html.H1(f"Image Embedding Explorer - {universe_name}",
                style={'textAlign': 'center', 'marginBottom': 30, 'color': COLORS['text']}),

        dcc.Tabs(id='tabs', value='location-search', children=[
            dcc.Tab(label='Location Similarity', value='location-search', className='tab'),
            dcc.Tab(label='Text Search', value='text-search', className='tab'),
            dcc.Tab(label='Change Detection', value='change-search', className='tab'),
            dcc.Tab(label='Statistics', value='stats', className='tab'),
        ]),

        html.Div(id='tab-content', style={'padding': 20})
    ], style={'maxWidth': '1200px', 'margin': '0 auto'})
