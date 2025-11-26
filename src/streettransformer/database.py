"""Database connection management for StreetTransformer."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

from .config import Config


@contextmanager
def get_connection(
    database_path: str | Path,
    read_only: bool = False
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Get a DuckDB connection with VSS extension loaded.

    Args:
        database_path: Path to DuckDB database file
        read_only: Whether to open in read-only mode

    Yields:
        DuckDB connection with spatial and VSS extensions loaded

    Example:
        >>> with get_connection("data.db") as con:
        ...     results = con.execute("SELECT * FROM lion.image_embeddings").df()
    """
    con = duckdb.connect(str(database_path), read_only=read_only)
    try:
        # Load spatial extension (for geometry support)
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")

        # Load VSS extension (for vector similarity search)
        con.execute("INSTALL vss;")
        con.execute("LOAD vss;")

        # Enable HNSW persistence
        con.execute("SET hnsw_enable_experimental_persistence = true;")

        yield con
    finally:
        con.close()


class Database:
    """Database connection manager for StreetTransformer.

    Provides a convenient interface for getting database connections
    using configuration.

    Attributes:
        config: Configuration with database path

    Example:
        >>> config = Config(database_path="data.db", universe_name="lion")
        >>> db = Database(config)
        >>> with db.connect() as con:
        ...     results = con.execute("SELECT * FROM lion.image_embeddings").df()
    """

    def __init__(self, config: Config):
        """Initialize database manager.

        Args:
            config: Configuration with database path
        """
        self.config = config

    @contextmanager
    def connect(
        self,
        read_only: bool = False
    ) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a database connection.

        Args:
            read_only: Whether to open in read-only mode

        Yields:
            DuckDB connection
        """
        with get_connection(self.config.database_path, read_only=read_only) as con:
            yield con

    def execute_query(self, query: str, read_only: bool = True):
        """Execute a query and return results as DataFrame.

        Args:
            query: SQL query to execute
            read_only: Whether to use read-only connection

        Returns:
            Query results as pandas DataFrame

        Example:
            >>> db = Database(config)
            >>> df = db.execute_query("SELECT * FROM lion.image_embeddings LIMIT 10")
        """
        with self.connect(read_only=read_only) as con:
            return con.execute(query).df()
