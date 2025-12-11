"""Details panel helper functions."""

import logging

logger = logging.getLogger(__name__)


def get_location_details(location_id, query_year, state):
    """Get and format location details for display.

    Args:
        location_id: Location to get details for
        query_year: Year from the query (for image carousel)
        state: Application state module

    Returns:
        Tuple of (details_content, location_info) or (None, None) if error
    """
    from streettransformer.db.database import get_connection
    from . import (
        DetailsStatsViewer,
        DetailsImageViewer,
        DetailsDocumentViewer,
        DetailsProjectViewer
    )

    try:
        with get_connection(state.CONFIG.database_path, read_only=True) as con:
            # Get location info
            query = f"""
                SELECT
                    location_id,
                    COALESCE(
                        array_to_string(additional_streets, ', '),
                        CONCAT(street1, ' & ', street2)
                    ) as street_name
                FROM {state.CONFIG.universe_name}.locations
                WHERE location_id = '{location_id}'
            """
            result = con.execute(query).df()

            if result.empty:
                return None, None

            street_name = result.iloc[0]['street_name']

            # Get images
            image_query = f"""
                SELECT path, media_type, year
                FROM {state.CONFIG.universe_name}.media_embeddings
                WHERE location_id = '{location_id}'
                    AND media_type = 'image'
                    AND path IS NOT NULL
                ORDER BY year ASC
                LIMIT 5
            """
            images_df = con.execute(image_query).df()

            # Rename 'path' to 'image_path' for backward compatibility with viewer
            if not images_df.empty:
                images_df = images_df.rename(columns={'path': 'image_path'})

            # Create viewer instances with data and combine their content
            details = []

            # 1. Stats/Header Section
            stats_viewer = DetailsStatsViewer(location_id=location_id, street_name=street_name)
            details.extend(stats_viewer.content)

            # 2. Image Carousel Section
            image_viewer = DetailsImageViewer(images_df=images_df, query_year=query_year)
            details.extend(image_viewer.content)

            # 3. Document Section (stub)
            document_viewer = DetailsDocumentViewer(location_id=location_id)
            details.extend(document_viewer.content)

            # 4. Project Section (stub)
            project_viewer = DetailsProjectViewer(location_id=location_id)
            details.extend(project_viewer.content)

            return details, location_id

    except Exception as e:
        logger.error(f"Error getting location details: {e}", exc_info=True)
        return None, None
