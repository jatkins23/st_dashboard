"""Map utilities for location visualization."""

import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SAVED_DATA_PATH = Path('dashboard/data')

def _create_project_hover_text(df: pd.DataFrame, prefix: str = "") -> pd.Series:
    """Create hover text for project markers.

    Args:
        df: DataFrame with project data
        prefix: Optional prefix for hover text (e.g., "RESULT", "QUERY LOCATION")

    Returns:
        Series with formatted hover text
    """
    def format_row(row):
        parts = []
        if prefix:
            parts.append(f"<b>{prefix}</b><br>")
        parts.extend([
            f"<b>{row.get('Project Title', 'N/A')}</b><br>",
            f"Project ID: {row.get('ProjectID', 'N/A')}<br>",
            f"Lead Agency: {row.get('Lead Agency', 'N/A')}<br>",
            f"Year: {row.get('Project Year', 'N/A')}<br>",
            f"Status: {row.get('Project Status', 'N/A')}<br>",
            f"Safety Scope: {row.get('Safety Scope', 'N/A')}<br>",
            f"Total Scope: {row.get('Total Scope', 'N/A')}"
        ])
        return "".join(parts)

    return df.apply(format_row, axis=1)


def _add_project_trace(fig: go.Figure, df: pd.DataFrame, name: str, color: str,
                       size: int, opacity: float, prefix: str = "") -> None:
    """Add a project trace to the map figure.

    Args:
        fig: Plotly figure to add trace to
        df: DataFrame with project data
        name: Name for the trace
        color: Marker color
        size: Marker size
        opacity: Marker opacity
        prefix: Optional prefix for hover text
    """
    if df.empty:
        return

    hover_text = _create_project_hover_text(df, prefix)

    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=dict(
            size=size,
            color=color,
            opacity=opacity
        ),
        text=hover_text,
        hoverinfo='text',
        customdata=df['location_id'],
        name=name
    ))


def create_location_map(projects_df: pd.DataFrame,
                        selected_location_id: str = None,
                        result_location_ids: list = None,
                        center_lat: float = 40.7128,
                        center_lon: float = -74.0060,
                        zoom: int = 10) -> go.Figure:
    """Create a Plotly mapbox figure with location markers.

    Args:
        selected_location_id: ID of query/selected location (shown in red)
        result_location_ids: List of result location IDs (shown in blue)
        projects_df: DataFrame with project data for background layer
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Map zoom level

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Display projects instead of locations on the map
    if projects_df is not None and not projects_df.empty:
        # Separate projects into categories
        result_location_ids = result_location_ids or []

        # Query location projects (red)
        query_projects = projects_df[projects_df['location_id'] == selected_location_id] if selected_location_id else pd.DataFrame()

        # Result location projects (blue)
        result_projects = projects_df[projects_df['location_id'].isin(result_location_ids)] if result_location_ids else pd.DataFrame()

        # Other projects (dimmed orange)
        exclude_ids = []
        if selected_location_id is not None:
            exclude_ids.append(selected_location_id)
        exclude_ids.extend(result_location_ids)

        other_projects = projects_df[~projects_df['location_id'].isin(exclude_ids)] if exclude_ids else projects_df

        # Add traces in order: other -> results -> query (so query is on top)
        _add_project_trace(fig, other_projects, 'Projects', '#FFA500', size=6, opacity=0.3)
        _add_project_trace(fig, result_projects, 'Result Projects', '#1E90FF', size=10, opacity=0.9, prefix='RESULT')
        _add_project_trace(fig, query_projects, 'Query Project', '#f48771', size=14, opacity=1.0, prefix='QUERY LOCATION')

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        showlegend=False,
        hovermode='closest'
    )

    return fig


def load_location_coordinates(config, db_connection_func, year: int = None, limit: int = None) -> pd.DataFrame:
    """Load location coordinates from database.

    Args:
        config: STConfig object with database path and universe name
        db_connection_func: Function that returns database connection
        year: Optional year filter (if None, gets one representative point per location)
        limit: Maximum number of locations to load (None for all)

    Returns:
        DataFrame with location_id, latitude, longitude, location_key columns
    """
    with db_connection_func() as con:
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        if year is not None:
            # Get locations for specific year
            query = f"""
                SELECT DISTINCT
                    l.location_id,
                    ST_Y(l.geometry) as latitude,
                    ST_X(l.geometry) as longitude,
                    l.street1,
                    l.street2,
                    l.additional_streets,
                    COALESCE(
                        CASE
                            WHEN l.additional_streets IS NOT NULL
                            THEN array_to_string(l.additional_streets, ', ')
                            ELSE NULL
                        END,
                        CONCAT(l.street1, ' & ', l.street2)
                    ) as location_key
                FROM {config.universe_name}.locations l
                INNER JOIN {config.universe_name}.media_embeddings e
                    ON l.location_id = e.location_id
                WHERE e.year = {year}
                    AND l.geometry IS NOT NULL
                    AND e.embedding IS NOT NULL
                {limit_clause}
            """
        else:
            # Get one point per location (using most recent year available)
            query = f"""
                SELECT DISTINCT
                    l.location_id,
                    ST_Y(l.geometry) as latitude,
                    ST_X(l.geometry) as longitude,
                    l.street1,
                    l.street2,
                    l.additional_streets,
                    COALESCE(
                        CASE
                            WHEN l.additional_streets IS NOT NULL
                            THEN array_to_string(l.additional_streets, ', ')
                            ELSE NULL
                        END,
                        CONCAT(l.street1, ' & ', l.street2)
                    ) as location_key
                FROM {config.universe_name}.locations l
                WHERE EXISTS (
                    SELECT 1 FROM {config.universe_name}.media_embeddings e
                    WHERE e.location_id = l.location_id
                    AND e.embedding IS NOT NULL
                )
                {limit_clause}
            """
        df = con.execute(query).df()
        return df


def load_projects(db_connection_func, universe_name:str = 'nyc') -> pd.DataFrame:
    """Load projects for map display from universe.

    Args:
        db_connection_func: Function that returns database connection

    Returns:
        DataFrame with project data including geometry for mapping
    """

    with db_connection_func() as con:
        query = f"""
            SELECT
                p.citydata_proj_id as ProjectID,
                p.ProjTitle as "Project Title",
                p.LeadAgency as "Lead Agency",
                p.proj_year as "Project Year",
                p.safety_scope as "Safety Scope",
                p.total_scope as "Total Scope",
                p.ProjectStatus as "Project Status",
                l2p.location_id,
                ST_Y(l.geometry) as latitude,
                ST_X(l.geometry) as longitude
            FROM {universe_name}.projects p
            INNER JOIN {universe_name}._location_to_project l2p
                ON p.citydata_proj_id = l2p.citydata_proj_Id
            INNER JOIN {universe_name}.locations l
                ON l2p.location_id = l.location_id
        """
        df = con.execute(query).df()
        return df


def load_all_streets(config, db_connection_func) -> list:
    """Load all unique street names from locations.

    Args:
        config: STConfig object with database path and universe name
        db_connection_func: Function that returns database connection

    Returns:
        Sorted list of unique street names
    """
    # Try to load it from the saved data
    df = read_saved_data('street_list.csv', config.universe_name)
    
    # # If it doesn't exist, then load it from the db
    if df is None:
        logger.debug('Saved `street_list` not found.\n\tLoading from db')
        with db_connection_func() as con:
            # Get all street columns and unnest additional_streets array
            query = f"""
                WITH all_streets AS (
                    SELECT street1 as street FROM {config.universe_name}.locations WHERE street1 IS NOT NULL
                    UNION
                    SELECT street2 as street FROM {config.universe_name}.locations WHERE street2 IS NOT NULL
                    UNION
                    SELECT UNNEST(additional_streets) as street
                    FROM {config.universe_name}.locations
                    WHERE additional_streets IS NOT NULL
                )
                SELECT DISTINCT street
                FROM all_streets
                WHERE EXISTS (
                    SELECT 1 FROM {config.universe_name}.locations l
                    INNER JOIN {config.universe_name}.media_embeddings e
                        ON l.location_id = e.location_id
                    WHERE e.embedding IS NOT NULL
                    AND (l.street1 = all_streets.street
                        OR l.street2 = all_streets.street
                        OR list_contains(l.additional_streets, all_streets.street))
                )
                ORDER BY street
            """
            df = con.execute(query).df()
    street_list =  df['street'].dropna().to_list()
    #street_list = [f'street_{x}' for x in range(100)]
        
    return street_list


def load_all_boroughs(config, db_connection_func) -> list:
    """Load all unique borough names from locations.

    Args:
        config: STConfig object with database path and universe name
        db_connection_func: Function that returns database connection

    Returns:
        Sorted list of unique borough names
    """
    try:
        with db_connection_func() as con:
            query = f"""
                SELECT DISTINCT boro
                FROM {config.universe_name}.locations
                WHERE boro IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM {config.universe_name}.media_embeddings e
                    WHERE e.location_id = locations.location_id
                    AND e.embedding IS NOT NULL
                )
                ORDER BY boro
            """
            df = con.execute(query).df()
            return df['boro'].dropna().to_list()
    except Exception as e:
        logger.error(f"Failed to load boroughs: {e}")
        return []


def get_location_details(config, db_connection_func, location_id: str) -> dict:
    """Get detailed information about a specific location.

    Args:
        config: STConfig object
        db_connection_func: Function that returns database connection
        location_id: Location ID to get details for

    Returns:
        Dictionary with location details
    """
    with db_connection_func() as con:
        query = f"""
            SELECT
                l.location_id,
                l.location_key,
                ST_Y(l.geometry) as latitude,
                ST_X(l.geometry) as longitude,
                l.street1,
                l.street2,
                l.additional_streets,
                COUNT(DISTINCT e.year) as year_count,
                MIN(e.year) as first_year,
                MAX(e.year) as last_year,
                COUNT(*) as image_count
            FROM {config.universe_name}.locations l
            LEFT JOIN {config.universe_name}.media_embeddings e
                ON l.location_id = e.location_id
            WHERE l.location_id = {location_id}
            GROUP BY l.location_id, l.location_key, l.geometry, l.street1, l.street2, l.additional_streets
        """

        result = con.execute(query).df()

        if result.empty:
            return None

        row = result.iloc[0]
        return {
            'location_id': int(row['location_id']),
            'location_key': str(row['location_key']) if pd.notna(row['location_key']) else None,
            'latitude': float(row['latitude']) if pd.notna(row['latitude']) else None,
            'longitude': float(row['longitude']) if pd.notna(row['longitude']) else None,
            'street1': str(row['street1']) if pd.notna(row['street1']) else None,
            'street2': str(row['street2']) if pd.notna(row['street2']) else None,
            'additional_streets': str(row['additional_streets']) if pd.notna(row['additional_streets']) else None,
            'year_count': int(row['year_count']),
            'first_year': int(row['first_year']) if pd.notna(row['first_year']) else None,
            'last_year': int(row['last_year']) if pd.notna(row['last_year']) else None,
            'image_count': int(row['image_count'])
        }

def read_saved_data(file_name: str, universe_name: str, saved_data_path: Path = SAVED_DATA_PATH): 
    file_path = saved_data_path / universe_name / file_name
    if not file_path.exists():
        return None
    
    match file_path.suffix:
        case '.csv':
            return pd.read_csv(file_path)
        case '.parquet':
            return pd.read_parquet(file_path)
        case '.txt':
            with open(file_path, 'r') as file:
                return file.read()
        case _:
            logger.warning(f'Unsure how to handle `{file_path}`')
            return file_path
        