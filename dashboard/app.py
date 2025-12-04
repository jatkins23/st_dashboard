"""
Simple dashboard application with Bootstrap UI.

Usage:
    python -m dashboard.simple_app --db /path/to/core.ddb --universe lion --port 8050
"""

import argparse
import logging

from streettransformer import STConfig, EmbeddingDB
from streettransformer.db.database import get_connection

from .frontend.layout import create_app, create_layout
#from frontend.layout import Layout
from .callbacks import register_all_callbacks
from .utils.map_utils import load_location_coordinates, load_projects
from . import state

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Image Embedding Dashboard")
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
    config = STConfig(
        database_path=args.db,
        universe_name=args.universe
    )
    db = EmbeddingDB(config)

    # Get available years
    stats = db.get_stats()
    available_years = sorted(stats['years'])

    # Load locations for the map
    logger.info("Loading location coordinates for map...")
    all_locations_df = load_location_coordinates(
        config,
        lambda: get_connection(config.database_path, read_only=True),
        limit=None
    )
    logger.info(f"Loaded {len(all_locations_df)} locations")

    # Load projects for map display
    logger.info("Loading projects for map...")
    try:
        projects_df = load_projects(
            lambda: get_connection(config.database_path, read_only=True),
            universe_name=config.universe_name
        )
        logger.info(f"Loaded {len(projects_df)} projects")
    except Exception as e:
        logger.warning(f"Could not load projects for {config.universe_name}: {e}")
        projects_df = None

    # Initialize global state
    state.initialize_state(config, db, available_years, all_locations_df, projects_df)

    # Create app and layout
    app = create_app()
    app.layout = create_layout(args.universe)

    # Register all callbacks
    register_all_callbacks(app)

    print(f"\n{'='*60}")
    print(f"Street Transformer Dashboard")
    print(f"{'='*60}")
    print(f"Universe: {args.universe}")
    print(f"Database: {args.db}")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Available Years: {available_years}")
    print(f"Locations Loaded: {len(all_locations_df)}")
    print(f"{'='*60}\n")

    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
