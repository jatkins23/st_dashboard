"""
Clean dashboard application with map-centric UI.

Usage:
    python -m dashboard.app_clean --db /path/to/core.ddb --universe lion --port 8050
"""

import argparse
import logging
from pathlib import Path

from dash import Input, Output, State, html
import pandas as pd

from streettransformer import Config, EmbeddingDB
from streettransformer.database import get_connection

from .backend.search import (
    search_by_location,
    search_by_text,
    search_change_patterns,
    get_embedding_stats
)
from .frontend.layout_v2 import create_app_with_styling, create_clean_layout
from .frontend.components.search_module import (
    create_search_bar,
    get_state_search_fields,
    get_text_search_fields,
    get_change_search_fields,
    get_stats_fields
)
from .frontend.components.results import format_results_accordion, format_change_results_accordion
from .frontend.components.details_module import render_simple_details
from .utils.map_utils import create_location_map, load_location_coordinates
from .config import COLORS

logger = logging.getLogger(__name__)

# Global state
CONFIG = None
DB = None
AVAILABLE_YEARS = []
ALL_LOCATIONS_DF = None


def get_location_basic_info(config, location_id: int):
    """Get basic info about a location."""
    with get_connection(config.database_path, read_only=True) as con:
        query = f"""
            SELECT
                l.location_id,
                COALESCE(
                    CASE
                        WHEN l.additional_streets IS NOT NULL
                        THEN array_to_string(l.additional_streets, ', ')
                        ELSE NULL
                    END,
                    CONCAT(l.street1, ' & ', l.street2)
                ) as street_name
            FROM {config.universe_name}.locations l
            WHERE l.location_id = {location_id}
        """
        result = con.execute(query).df()
        if result.empty:
            return None
        return result.iloc[0].to_dict()


def get_location_images(config, location_id: int, limit: int = 10):
    """Get images for a location."""
    with get_connection(config.database_path, read_only=True) as con:
        query = f"""
            SELECT image_path, year
            FROM {config.universe_name}.image_embeddings
            WHERE location_id = {location_id}
                AND image_path IS NOT NULL
            ORDER BY year DESC
            LIMIT {limit}
        """
        result = con.execute(query).df()
        return result.to_dict('records') if not result.empty else []


def setup_callbacks(app):
    """Setup all dashboard callbacks."""

    @app.callback(
        Output('search-bar-container', 'children'),
        Input('main-map', 'id')  # Dummy input to trigger on page load
    )
    def render_search_bar(_):
        """Render the search bar."""
        return create_search_bar(AVAILABLE_YEARS)

    @app.callback(
        Output('search-fields', 'children'),
        Input('search-tabs', 'value')
    )
    def update_search_fields(tab):
        """Update search fields based on selected tab."""
        if tab == 'state':
            return get_state_search_fields(AVAILABLE_YEARS)
        elif tab == 'text':
            return get_text_search_fields(AVAILABLE_YEARS)
        elif tab == 'change':
            return get_change_search_fields(AVAILABLE_YEARS)
        elif tab == 'stats':
            return get_stats_fields()
        return html.Div()

    @app.callback(
        Output('results-content', 'children'),
        Output('results-panel', 'style'),
        Output('result-locations', 'data'),
        Input('search-btn', 'n_clicks'),
        State('location-id-input', 'value'),
        State('year-dropdown', 'value'),
        State('target-year-dropdown', 'value'),
        State('limit-dropdown', 'value'),
        State('use-faiss-checkbox', 'value'),
        State('use-whitening-checkbox', 'value'),
        State('results-panel', 'style'),
        prevent_initial_call=True
    )
    def handle_state_search(n_clicks, location_id, year, target_year, limit, use_faiss, use_whitening, panel_style):
        """Handle state/location search."""
        if not location_id or not year:
            return html.Div("Please enter location ID and year", className='error-message'), panel_style, []

        try:
            use_faiss_enabled = 'faiss' in (use_faiss or [])
            use_whitening_enabled = 'whitening' in (use_whitening or [])

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
                return (
                    html.Div(f"No results found", className='warning-message'),
                    {**panel_style, 'display': 'block'},
                    []
                )

            result_location_ids = results['location_id'].tolist()

            return (
                html.Div([
                    html.H4(f"Top {len(results)} similar locations:",
                           style={'color': COLORS['success'], 'marginBottom': 15, 'fontSize': '16px'}),
                    format_results_accordion(results)
                ]),
                {**panel_style, 'display': 'block'},
                result_location_ids
            )

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return (
                html.Div(f"Error: {str(e)}", className='error-message'),
                {**panel_style, 'display': 'block'},
                []
            )

    @app.callback(
        Output('details-content', 'children'),
        Output('details-panel', 'style'),
        Output('query-location-id', 'data'),
        Input('location-id-input', 'value'),
        Input('change-location-id-input', 'value'),
        Input('main-map', 'clickData'),
        State('details-panel', 'style'),
        prevent_initial_call=False
    )
    def update_details(state_location_id, change_location_id, click_data, panel_style):
        """Update details panel when location is selected."""
        location_id = state_location_id or change_location_id

        # Handle map click
        if click_data and 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            if 'customdata' in point:
                location_id = point['customdata']

        if not location_id:
            return (
                html.Div("Enter a location ID or click on the map",
                        style={'color': COLORS['text-secondary'], 'fontStyle': 'italic',
                               'padding': '20px', 'textAlign': 'center'}),
                {**panel_style, 'display': 'none'},
                None
            )

        try:
            # Get location info
            location_info = get_location_basic_info(CONFIG, location_id)
            if not location_info:
                return (
                    html.Div(f"Location {location_id} not found",
                            style={'color': COLORS['error'], 'padding': '20px'}),
                    {**panel_style, 'display': 'block'},
                    location_id
                )

            # Get images
            images = get_location_images(CONFIG, location_id)

            return (
                render_simple_details(location_id, location_info['street_name'], images),
                {**panel_style, 'display': 'block'},
                location_id
            )

        except Exception as e:
            logger.error(f"Error getting location details: {e}", exc_info=True)
            return (
                html.Div(f"Error: {str(e)}", style={'color': COLORS['error'], 'padding': '20px'}),
                {**panel_style, 'display': 'block'},
                location_id
            )

    @app.callback(
        Output('main-map', 'figure'),
        Input('query-location-id', 'data'),
        Input('result-locations', 'data'),
        prevent_initial_call=False
    )
    def update_map(query_location_id, result_location_ids):
        """Update map with selected location and results."""
        global ALL_LOCATIONS_DF

        # NYC coordinates
        center_lat = 40.7128
        center_lon = -74.0060

        # Create map with Positron Dark style
        fig = create_location_map(
            locations_df=ALL_LOCATIONS_DF,
            selected_location_id=query_location_id,
            result_location_ids=result_location_ids or [],
            center_lat=center_lat,
            center_lon=center_lon,
            zoom=11
        )

        return fig


def main():
    """Main entry point."""
    global CONFIG, DB, AVAILABLE_YEARS, ALL_LOCATIONS_DF

    parser = argparse.ArgumentParser(description="Image Embedding Dashboard (Clean)")
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

    # Load all locations for the map
    logger.info("Loading location coordinates for map...")
    ALL_LOCATIONS_DF = load_location_coordinates(
        CONFIG,
        lambda: get_connection(CONFIG.database_path, read_only=True),
        limit=5000
    )
    logger.info(f"Loaded {len(ALL_LOCATIONS_DF)} locations")

    # Create app
    app, index_string = create_app_with_styling()
    app.index_string = index_string
    app.layout = create_clean_layout(args.universe)

    # Setup callbacks
    setup_callbacks(app)

    print(f"\n{'='*60}")
    print(f"Image Embedding Dashboard (Clean)")
    print(f"{'='*60}")
    print(f"Universe: {args.universe}")
    print(f"Database: {args.db}")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Available Years: {AVAILABLE_YEARS}")
    print(f"Locations Loaded: {len(ALL_LOCATIONS_DF)}")
    print(f"{'='*60}\n")

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
