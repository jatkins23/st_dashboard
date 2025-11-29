"""Map utilities for location visualization."""

import plotly.graph_objects as go
import pandas as pd
from ..config import COLORS


def create_location_map(locations_df: pd.DataFrame = None,
                        selected_location_id: int = None,
                        result_location_ids: list = None,
                        center_lat: float = 40.7128,
                        center_lon: float = -74.0060,
                        zoom: int = 10) -> go.Figure:
    """Create a Plotly mapbox figure with location markers.

    Args:
        locations_df: DataFrame with columns: location_id, latitude, longitude, location_key
        selected_location_id: ID of query/selected location (shown in red)
        result_location_ids: List of result location IDs (shown in blue)
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Map zoom level

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if locations_df is not None and not locations_df.empty:
        # Separate locations into categories
        result_location_ids = result_location_ids or []

        # Query location (red)
        if selected_location_id is not None:
            query_loc = locations_df[locations_df['location_id'] == selected_location_id]
        else:
            query_loc = pd.DataFrame()

        # Result locations (blue)
        if result_location_ids:
            result_locs = locations_df[locations_df['location_id'].isin(result_location_ids)]
        else:
            result_locs = pd.DataFrame()

        # Other locations (dimmed gray)
        exclude_ids = []
        if selected_location_id is not None:
            exclude_ids.append(selected_location_id)
        exclude_ids.extend(result_location_ids)

        if exclude_ids:
            other_locs = locations_df[~locations_df['location_id'].isin(exclude_ids)]
        else:
            other_locs = locations_df

        # Add other locations (dimmed gray)
        if not other_locs.empty:
            hover_text = other_locs.apply(
                lambda row: f"ID: {row['location_id']}<br>{row.get('location_key', '')}",
                axis=1
            )

            fig.add_trace(go.Scattermapbox(
                lat=other_locs['latitude'],
                lon=other_locs['longitude'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='#555555',
                    opacity=0.3
                ),
                text=hover_text,
                hoverinfo='text',
                customdata=other_locs['location_id'],
                name='Other Locations'
            ))

        # Add result locations (blue)
        if not result_locs.empty:
            hover_text = result_locs.apply(
                lambda row: f"<b>RESULT</b><br>ID: {row['location_id']}<br>{row.get('location_key', '')}",
                axis=1
            )

            fig.add_trace(go.Scattermapbox(
                lat=result_locs['latitude'],
                lon=result_locs['longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#1E90FF',  # Dodger blue
                    opacity=0.9
                ),
                text=hover_text,
                hoverinfo='text',
                customdata=result_locs['location_id'],
                name='Results'
            ))

        # Add query location (red) - drawn last so it's on top
        if not query_loc.empty:
            hover_text = query_loc.apply(
                lambda row: f"<b>QUERY LOCATION</b><br>ID: {row['location_id']}<br>{row.get('location_key', '')}",
                axis=1
            )

            fig.add_trace(go.Scattermapbox(
                lat=query_loc['latitude'],
                lon=query_loc['longitude'],
                mode='markers',
                marker=dict(
                    size=14,
                    color=COLORS['error'],  # Red for query location
                    opacity=1.0,
                    symbol='circle'
                ),
                text=hover_text,
                hoverinfo='text',
                customdata=query_loc['location_id'],
                name='Query Location'
            ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=800,
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        showlegend=False,
        hovermode='closest'
    )

    return fig


def load_location_coordinates(config, db_connection_func, year: int = None, limit: int = 5000) -> pd.DataFrame:
    """Load location coordinates from database.

    Args:
        config: STConfig object with database path and universe name
        db_connection_func: Function that returns database connection
        year: Optional year filter (if None, gets one representative point per location)
        limit: Maximum number of locations to load

    Returns:
        DataFrame with location_id, latitude, longitude, location_key columns
    """
    with db_connection_func() as con:
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
                INNER JOIN {config.universe_name}.image_embeddings e
                    ON l.location_id = e.location_id
                WHERE e.year = {year}
                    AND l.geometry IS NOT NULL
                    AND e.embedding IS NOT NULL
                LIMIT {limit}
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
                    SELECT 1 FROM {config.universe_name}.image_embeddings e
                    WHERE e.location_id = l.location_id
                    AND e.embedding IS NOT NULL
                )
                LIMIT {limit}
            """
        df = con.execute(query).df()
        return df


def get_location_details(config, db_connection_func, location_id: int) -> dict:
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
            LEFT JOIN {config.universe_name}.image_embeddings e
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
