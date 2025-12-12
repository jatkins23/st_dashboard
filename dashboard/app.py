"""
Simple dashboard application with Bootstrap UI.

Usage:
    python -m dashboard.simple_app --db /path/to/core.ddb --universe lion --port 8050
"""

import argparse
import logging
from pathlib import Path

from streettransformer import STConfig, EmbeddingDB
from streettransformer.db.database import get_connection
from dash import Dash
import dash_bootstrap_components as dbc
from .components.dashboard import Dashboard

from .utils.map_utils import load_location_coordinates, load_projects, load_all_streets, load_all_boroughs
from . import state

logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    import os

    parser = argparse.ArgumentParser(description="Simple Image Embedding Dashboard")
    parser.add_argument(
        '--db',
        type=str,
        default=os.getenv('ST_DATABASE_PATH'),
        help='Path to DuckDB database (default: ST_DATABASE_PATH env var)'
    )
    parser.add_argument(
        '--universe', '-u',
        type=str,
        default='nyc',
        help='Universe name (default: nyc)'
    )
    parser.add_argument('--port', '-p', type=int, default=8050, help='Port to run on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    # PostgreSQL vector search arguments
    from dotenv import load_dotenv
    load_dotenv()
    parser.add_argument('--pg-host', type=str, default=os.getenv('PG_HOST', 'localhost'),
                        help='PostgreSQL host (default: PG_HOST env var or localhost)')
    parser.add_argument('--pg-port', type=int, default=int(os.getenv('PGPORT', '5432')),
                        help='PostgreSQL port (default: PG_PORT env var or 5432)')
    parser.add_argument('--pg-db', type=str, default=os.getenv('PG_DATABASE', 'image_retrieval'),
                        help='PostgreSQL database name (default: PG_DATABASE env var or image_retrieval)')
    parser.add_argument('--pg-user', type=str, default=os.getenv('PG_USER', 'postgres'),
                        help='PostgreSQL user (default: PG_USER env var or postgres)')
    parser.add_argument('--pg-password', type=str, default=os.getenv('PG_PASSWORD', ''),
                        help='PostgreSQL password (default: PG_PASSWORD env var or empty)')
    parser.add_argument('--enable-vector-search', action='store_true', default=True,
                        help='Enable PostgreSQL vector search (default: enabled)')
    parser.add_argument('--vector-dim', type=int, default=512,
                        help='Vector dimension (default: 512 for CLIP ViT-B-32)')

    # Artifacts directory for FAISS and whitening
    parser.add_argument('--artifacts', type=str, default=None,
                        help='Path to artifacts directory containing FAISS indexes and whitening parameters')
    parser.add_argument('--enable-whitening', action='store_true', default=True,
                        help='Enable whitening reranking (default: enabled)')
    parser.add_argument('--enable-faiss', action='store_true', default=False,
                        help='Enable FAISS approximate search (default: disabled)')

    args = parser.parse_args()

    # Validate required arguments
    if args.db is None:
        parser.error("--db is required (or set ST_DATABASE_PATH environment variable)")
    if args.universe is None:
        parser.error("--universe is required")

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

    # Load all unique streets for the street selector
    logger.info("Loading street names...")
    all_streets = load_all_streets(
        config,
        lambda: get_connection(config.database_path, read_only=True)
    )
    logger.info(f"Loaded {len(all_streets)} unique streets")

    # Load all unique boroughs for the borough selector
    logger.info("Loading borough names...")
    all_boroughs = load_all_boroughs(
        config,
        lambda: get_connection(config.database_path, read_only=True)
    )
    logger.info(f"Loaded {len(all_boroughs)} unique boroughs")

    # Initialize global state
    state.initialize_state(config, db, available_years, all_locations_df, projects_df)

    # Initialize PostgreSQL vector search if enabled
    from dashboard.config import PGConfig, FeatureFlags

    pg_config = PGConfig(
        db_name=args.pg_db,
        db_user=args.pg_user,
        db_password=args.pg_password,
        db_host=args.pg_host,
        db_port=args.pg_port,
        vector_dimension=args.vector_dim,
        artifacts_dir=args.artifacts
    )

    feature_flags = FeatureFlags(
        use_vector_search=args.enable_vector_search,
        use_whitening=args.enable_whitening,
        use_faiss=args.enable_faiss
    )
    state.initialize_feature_flags(feature_flags)

    if args.enable_vector_search:
        logger.info("Initializing PostgreSQL vector search...")
        if state.initialize_postgres_pool(pg_config):
            if state.initialize_vector_db(pg_config):
                logger.info("Vector search enabled successfully")
            else:
                logger.warning("Vector search initialization failed, will use DuckDB fallback")
                feature_flags.use_vector_search = False
        else:
            logger.warning("PostgreSQL connection pool initialization failed, will use DuckDB fallback")
            feature_flags.use_vector_search = False

    # Create app and layout
    assets_folder = Path(__file__).parent.parent / 'assets'
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=str(assets_folder)
    )

    # Create dashboard (which creates all component instances internally)
    dashboard = Dashboard(
        universe_name=args.universe,
        available_years=available_years,
        all_streets=all_streets,
        all_boroughs=all_boroughs
    )

    app.layout = dashboard.layout

    # Register all component callbacks
    dashboard.register_callbacks(app)

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
