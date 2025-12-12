"""Global application state."""
import logging
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global state shared across callbacks
CONFIG = None
DB = None
AVAILABLE_YEARS = []
ALL_LOCATIONS_DF = None
PROJECTS_DF = None

# PostgreSQL vector search state
PG_POOL = None          # ThreadedConnectionPool for PostgreSQL
VECTOR_DB = None        # VectorDB instance (uses pool)
ENCODER_MANAGER = None  # EncoderManager singleton
FEATURE_FLAGS = None    # FeatureFlags instance
PG_CONFIG = None        # PGConfig instance


def initialize_state(config, db, available_years, locations_df, projects_df):
    """Initialize global state.

    Args:
        config: STConfig instance
        db: EmbeddingDB instance
        available_years: List of available years
        locations_df: DataFrame with location coordinates
        projects_df: DataFrame with projects data (optional)
    """
    global CONFIG, DB, AVAILABLE_YEARS, ALL_LOCATIONS_DF, PROJECTS_DF
    CONFIG = config
    DB = db
    AVAILABLE_YEARS = available_years
    ALL_LOCATIONS_DF = locations_df
    PROJECTS_DF = projects_df


def initialize_postgres_pool(pg_config):
    """Initialize PostgreSQL connection pool.

    Args:
        pg_config: PGConfig instance

    Returns:
        bool: True if initialization successful, False otherwise
    """
    global PG_POOL, PG_CONFIG
    try:
        from psycopg2.pool import ThreadedConnectionPool

        PG_CONFIG = pg_config
        PG_POOL = ThreadedConnectionPool(
            pg_config.min_connections,
            pg_config.max_connections,
            dbname=pg_config.db_name,
            user=pg_config.db_user,
            password=pg_config.db_password,
            host=pg_config.db_host,
            port=pg_config.db_port
        )
        logger.info(f"PostgreSQL connection pool initialized (host={pg_config.db_host}, db={pg_config.db_name})")
        return True
    except ImportError:
        logger.error("psycopg2 not installed. Install with: uv pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL pool: {e}")
        return False


@contextmanager
def get_pg_connection():
    """Get connection from pool with context manager.

    Yields:
        psycopg2 connection

    Raises:
        RuntimeError: If pool not initialized
    """
    if PG_POOL is None:
        raise RuntimeError("PostgreSQL pool not initialized. Call initialize_postgres_pool() first.")

    conn = PG_POOL.getconn()
    try:
        yield conn
    finally:
        PG_POOL.putconn(conn)


def initialize_vector_db(pg_config):
    """Initialize VectorDB instance using the connection pool.

    Args:
        pg_config: PGConfig instance

    Returns:
        bool: True if initialization successful, False otherwise
    """
    global VECTOR_DB
    try:
        from streettransformer.image_retrieval.vector_db import VectorDB

        if PG_POOL is None:
            logger.error("Cannot initialize VectorDB: connection pool not initialized")
            return False

        # VectorDB will use connection from pool
        VECTOR_DB = VectorDB(
            vector_dimension=pg_config.vector_dimension,
            dbname=pg_config.db_name,
            user=pg_config.db_user,
            password=pg_config.db_password,
            host=pg_config.db_host,
            port=pg_config.db_port
        )
        logger.info("VectorDB instance initialized")
        return True
    except ImportError as e:
        logger.error(f"Failed to import VectorDB: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize VectorDB: {e}")
        return False


def initialize_feature_flags(feature_flags):
    """Initialize feature flags.

    Args:
        feature_flags: FeatureFlags instance
    """
    global FEATURE_FLAGS
    FEATURE_FLAGS = feature_flags
    logger.info(f"Feature flags initialized: vector_search={feature_flags.use_vector_search}")


def cleanup_postgres():
    """Clean up PostgreSQL resources."""
    global PG_POOL, VECTOR_DB

    if VECTOR_DB is not None:
        try:
            VECTOR_DB.close()
            VECTOR_DB = None
            logger.info("VectorDB closed")
        except Exception as e:
            logger.error(f"Error closing VectorDB: {e}")

    if PG_POOL is not None:
        try:
            PG_POOL.closeall()
            PG_POOL = None
            logger.info("PostgreSQL connection pool closed")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}")
