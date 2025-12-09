"""Search-related helper functions."""

import logging

logger = logging.getLogger(__name__)


def get_location_from_streets(selected_streets, state):
    """Get location_id from selected street names.

    Args:
        selected_streets: List of selected street names
        state: Application state module

    Returns:
        location_id if found, None otherwise
    """
    from streettransformer.db.database import get_connection

    if not selected_streets or len(selected_streets) == 0:
        return None

    try:
        with get_connection(state.CONFIG.database_path, read_only=True) as con:
            # Find locations that match ALL selected streets
            street_conditions = []
            for street in selected_streets:
                street_conditions.append(f"""
                    (street1 = '{street}'
                     OR street2 = '{street}'
                     OR list_contains(additional_streets, '{street}'))
                """)

            where_clause = " AND ".join(street_conditions)

            query = f"""
                SELECT location_id
                FROM {state.CONFIG.universe_name}.locations
                WHERE {where_clause}
                LIMIT 1
            """
            result = con.execute(query).df()

            if not result.empty:
                return result.iloc[0]['location_id']
    except Exception as e:
        logger.error(f"Error finding location by streets: {e}", exc_info=True)

    return None


def execute_image_search(location_id, year, target_year, limit, media_type, state):
    """Execute the image-to-image search query.

    Args:
        location_id: Location to search from
        year: Year of the query image
        target_year: Optional target year filter
        limit: Maximum number of results
        media_type: Type of media to search
        state: Application state module

    Returns:
        QueryResultsSet with enriched results
    """
    from streettransformer.db.database import get_connection
    from streettransformer.query.queries.ask import ImageToImageStateQuery

    # Configuration settings
    use_faiss_enabled = True
    use_whitening_enabled = False

    # Default to 'image' if no media type selected
    selected_media_type = media_type if media_type else 'image'

    # Create and execute query
    query = ImageToImageStateQuery(
        config=state.CONFIG,
        db=state.DB,
        location_id=location_id,
        year=year,
        target_years=[target_year] if target_year else None,
        limit=limit,
        media_types=[selected_media_type],
        use_faiss=use_faiss_enabled,
        use_whitening=use_whitening_enabled,
        remove_self=True
    )

    results_set = query.search()

    # Enrich results with street names and image paths
    if len(results_set) > 0:
        with get_connection(state.CONFIG.database_path, read_only=True) as con:
            for result in results_set:
                result.enrich_street_names(con, state.CONFIG.universe_name)
                result.enrich_image_path(con, state.CONFIG.universe_name, selected_media_type)

    return results_set
