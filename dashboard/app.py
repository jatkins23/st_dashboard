"""
Complete dashboard application with all callbacks.

Usage:
    python -m dashboard.app --db /path/to/core.ddb --universe lion --port 8050
"""

import argparse
import logging
from pathlib import Path

from dash import Input, Output, State, html
import pandas as pd
import numpy as np

from streettransformer import Config, EmbeddingDB
from streettransformer.database import get_connection

from .backend.search import (
    search_by_location,
    search_by_text,
    search_change_patterns,
    get_embedding_stats
)
from .frontend.layout import create_app_with_styling, create_layout
from .frontend.tabs import (
    create_location_search_tab,
    create_text_search_tab,
    create_change_search_tab,
    create_stats_tab
)
from .frontend.components import format_results_accordion, format_change_results_accordion
from .config import COLORS

logger = logging.getLogger(__name__)

# Global state (set via CLI args)
CONFIG = None
DB = None
AVAILABLE_YEARS = []


def setup_callbacks(app):
    """Setup all Dash callbacks.

    Args:
        app: Dash app instance
    """

    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab(tab):
        """Render the appropriate tab based on selection."""
        if tab == 'location-search':
            return create_location_search_tab(AVAILABLE_YEARS)
        elif tab == 'text-search':
            return create_text_search_tab(AVAILABLE_YEARS)
        elif tab == 'change-search':
            return create_change_search_tab(AVAILABLE_YEARS)
        elif tab == 'stats':
            return create_stats_tab()
        else:
            return html.Div("Tab not implemented yet")

    # Location search callbacks
    @app.callback(
        Output('location-results', 'children'),
        Input('location-search-btn', 'n_clicks'),
        State('location-id-input', 'value'),
        State('year-dropdown', 'value'),
        State('target-year-dropdown', 'value'),
        State('limit-dropdown', 'value'),
        State('use-faiss-checkbox', 'value'),
        State('use-whitening-checkbox', 'value'),
        prevent_initial_call=True
    )
    def handle_location_search(n_clicks, location_id, year, target_year, limit, use_faiss, use_whitening):
        """Handle location similarity search."""
        # Validate inputs
        if isinstance(location_id, (list, tuple, np.ndarray, pd.Series)):
            return html.Div("Please enter a single location ID (not an array)", className='error-message')

        if location_id is None:
            return html.Div("Please enter a location ID", className='error-message')

        if isinstance(year, (list, tuple, np.ndarray, pd.Series)):
            return html.Div("Please select a single year (not an array)", className='error-message')

        if year is None:
            return html.Div("Please select a year", className='error-message')

        try:
            # Parse checkbox values
            use_faiss_enabled = 'faiss' in (use_faiss or [])
            use_whitening_enabled = 'whitening' in (use_whitening or [])

            # Search
            results = search_by_location(
                config=CONFIG,
                db=DB,
                db_connection_func=lambda: get_connection(CONFIG.database_path, read_only=True),
                location_id=location_id,
                year=year,
                target_year=target_year,
                limit=limit,
                use_faiss=use_faiss_enabled,
                use_whitening=use_whitening_enabled
            )

            if results.empty:
                return html.Div(
                    f"No results found for location {location_id}, year {year}",
                    className='warning-message'
                )

            # Build method info message
            methods_used = []
            if use_faiss_enabled:
                methods_used.append("FAISS")
            if use_whitening_enabled:
                methods_used.append("Whitening reranking")
            methods_msg = f" (using {', '.join(methods_used)})" if methods_used else ""

            return html.Div([
                html.H4(f"Top {len(results)} similar locations{methods_msg}:",
                       style={'color': COLORS['success'], 'marginBottom': 20}),
                format_results_accordion(results)
            ])

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", className='error-message')

    # Text search callbacks
    @app.callback(
        Output('text-results', 'children'),
        Input('text-search-btn', 'n_clicks'),
        State('text-query-input', 'value'),
        State('text-year-dropdown', 'value'),
        State('text-limit-dropdown', 'value'),
        State('text-use-faiss-checkbox', 'value'),
        State('text-use-whitening-checkbox', 'value'),
        prevent_initial_call=True
    )
    def handle_text_search(n_clicks, text_query, year, limit, use_faiss, use_whitening):
        """Handle text-to-image search."""
        if not text_query or not text_query.strip():
            return html.Div("Please enter a text query", className='error-message')

        try:
            # Parse checkbox values
            use_faiss_enabled = 'faiss' in (use_faiss or [])
            use_whitening_enabled = 'whitening' in (use_whitening or [])

            # Search
            results = search_by_text(
                config=CONFIG,
                db=DB,
                db_connection_func=lambda: get_connection(CONFIG.database_path, read_only=True),
                text_query=text_query,
                year=year,
                limit=limit,
                use_faiss=use_faiss_enabled,
                use_whitening=use_whitening_enabled
            )

            if results.empty:
                return html.Div(
                    f"No results found for query '{text_query}'",
                    className='warning-message'
                )

            # Build method info message
            methods_used = []
            if use_faiss_enabled:
                methods_used.append("FAISS")
            if use_whitening_enabled:
                methods_used.append("Whitening reranking")
            methods_msg = f" (using {', '.join(methods_used)})" if methods_used else ""

            year_msg = f" in year {year}" if year else ""

            return html.Div([
                html.H4(f"Top {len(results)} matches for \"{text_query}\"{year_msg}{methods_msg}:",
                       style={'color': COLORS['success'], 'marginBottom': 20}),
                format_results_accordion(results, show_years=(year is None))
            ])

        except Exception as e:
            logger.error(f"Text search error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", className='error-message')

    # Change search callbacks
    @app.callback(
        Output('change-results', 'children'),
        Input('change-search-btn', 'n_clicks'),
        State('change-location-id-input', 'value'),
        State('year-from-dropdown', 'value'),
        State('year-to-dropdown', 'value'),
        State('change-limit-dropdown', 'value'),
        prevent_initial_call=True
    )
    def handle_change_search(n_clicks, location_id, year_from, year_to, limit):
        """Handle change pattern detection search."""
        if not location_id:
            return html.Div("Please enter a location ID", className='error-message')

        if not all([year_from, year_to]):
            return html.Div("Please select both years", className='error-message')

        try:
            results = search_change_patterns(
                config=CONFIG,
                db=DB,
                db_connection_func=lambda: get_connection(CONFIG.database_path, read_only=True),
                location_id=location_id,
                year_from=year_from,
                year_to=year_to,
                limit=limit
            )

            if results.empty:
                return html.Div(
                    f"No results found for location {location_id} change pattern",
                    className='warning-message'
                )

            return html.Div([
                html.H4(
                    f"Locations with similar changes ({year_from} → {year_to}):",
                    style={'color': COLORS['success'], 'marginBottom': 20}
                ),
                format_change_results_accordion(results)
            ])

        except Exception as e:
            logger.error(f"Change search error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", className='error-message')

    # Statistics callback
    @app.callback(
        Output('stats-content', 'children'),
        Input('stats-refresh-btn', 'n_clicks'),
        prevent_initial_call=False
    )
    def handle_stats(n_clicks):
        """Handle statistics display."""
        try:
            stats = get_embedding_stats(DB)

            return html.Div([
                html.H4("Database Statistics"),
                html.Div([
                    html.Div([
                        html.Strong("Total Embeddings: "),
                        html.Span(f"{stats['total_embeddings']:,}")
                    ], style={'marginBottom': 10}),
                    html.Div([
                        html.Strong("Number of Years: "),
                        html.Span(str(stats['year_count']))
                    ], style={'marginBottom': 10}),
                    html.Div([
                        html.Strong("Years Available: "),
                        html.Span(str(stats['years']))
                    ], style={'marginBottom': 10}),
                    html.Div([
                        html.Strong("Vector Dimension: "),
                        html.Span(str(stats['vector_dim']))
                    ], style={'marginBottom': 10}),
                    html.Div([
                        html.Strong("Universe: "),
                        html.Span(stats['universe_name'])
                    ], style={'marginBottom': 10}),
                ], style={'fontSize': 16})
            ])

        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", className='error-message')


def main():
    """Main entry point for the dashboard."""
    global CONFIG, DB, AVAILABLE_YEARS

    parser = argparse.ArgumentParser(description="Image Embedding Dashboard")
    parser.add_argument('--db', type=str, required=True, help='Path to DuckDB database')
    parser.add_argument('--universe', '-u', type=str, required=True, help='Universe name')
    parser.add_argument('--port', '-p', type=int, default=8050, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize config and database
    CONFIG = Config(
        database_path=args.db,
        universe_name=args.universe
    )
    DB = EmbeddingDB(CONFIG)

    # Get available years
    stats = DB.get_stats()
    AVAILABLE_YEARS = sorted(stats['years'])

    logger.info(f"Starting dashboard for universe: {args.universe}")
    logger.info(f"Available years: {AVAILABLE_YEARS}")

    # Create app
    app, index_string = create_app_with_styling()
    app.index_string = index_string
    app.layout = create_layout(args.universe)

    # Setup callbacks
    setup_callbacks(app)

    print(f"\n{'='*60}")
    print(f"Image Embedding Dashboard")
    print(f"{'='*60}")
    print(f"Universe: {args.universe}")
    print(f"Database: {args.db}")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Available Years: {AVAILABLE_YEARS}")
    print(f"{'='*60}\n")

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()