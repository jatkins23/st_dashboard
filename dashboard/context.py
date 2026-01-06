"""Global application state."""

# Global state shared across callbacks
CONFIG = None
DB = None
AVAILABLE_YEARS = []
ALL_LOCATIONS_DF = None
PROJECTS_DF = None


def initialize_context(config, db, available_years, locations_df, projects_df):
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
