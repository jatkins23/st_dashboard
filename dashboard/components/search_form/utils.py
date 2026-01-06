"""Shared utilities for search forms."""

import logging
from streettransformer.db.database import get_connection

logger = logging.getLogger(__name__)


def filter_street_options_by_selection(selected_streets, current_data, app_ctx):
    """Filter street options to only show valid combinations.

    When streets are selected, only show streets that appear in locations
    that also have ALL the currently selected streets.

    Args:
        selected_streets: List of currently selected street names
        current_data: Current dropdown data
        app_ctx: Application context module with CONFIG

    Returns:
        List of filtered street options in format [{"label": str, "value": str}]
    """
    # If no streets selected, return all streets
    if not selected_streets or len(selected_streets) == 0:
        logger.info("No streets selected, returning all streets")
        return current_data

    try:
        logger.info(f"Filtering streets for selection: {selected_streets}")
        with get_connection(app_ctx.CONFIG.database_path, read_only=True) as con:
            # Build conditions to find locations with ALL selected streets
            street_conditions = []
            for street in selected_streets:
                street_conditions.append(f"""
                    (street1 = '{street}'
                     OR street2 = '{street}'
                     OR list_contains(additional_streets, '{street}'))
                """)

            where_clause = " AND ".join(street_conditions)

            # Get all streets from locations that match the selected streets
            query = f"""
                WITH matching_locations AS (
                    SELECT location_id
                    FROM {app_ctx.CONFIG.universe_name}.locations
                    WHERE {where_clause}
                ),
                all_streets AS (
                    SELECT street1 as street
                    FROM {app_ctx.CONFIG.universe_name}.locations
                    WHERE location_id IN (SELECT location_id FROM matching_locations)
                      AND street1 IS NOT NULL
                    UNION
                    SELECT street2 as street
                    FROM {app_ctx.CONFIG.universe_name}.locations
                    WHERE location_id IN (SELECT location_id FROM matching_locations)
                      AND street2 IS NOT NULL
                    UNION
                    SELECT UNNEST(additional_streets) as street
                    FROM {app_ctx.CONFIG.universe_name}.locations
                    WHERE location_id IN (SELECT location_id FROM matching_locations)
                      AND additional_streets IS NOT NULL
                )
                SELECT DISTINCT street
                FROM all_streets
                ORDER BY street
            """

            df = con.execute(query).df()
            valid_streets = df['street'].dropna().tolist()

            logger.info(f"Found {len(valid_streets)} valid cross-streets")
            if len(valid_streets) > 0:
                logger.info(f"Sample streets: {valid_streets[:5]}")

            # Return filtered options
            return [{"label": s, "value": s} for s in valid_streets]

    except Exception as e:
        logger.error(f"Error filtering street options: {e}", exc_info=True)
        # On error, return current data
        return current_data


def get_location_from_streets(selected_streets, app_ctx):
    """Get location_id from selected street names.

    Args:
        selected_streets: List of selected street names
        app_ctx: Application context module

    Returns:
        location_id if found, None otherwise
    """
    if not selected_streets or len(selected_streets) == 0:
        return None

    try:
        with get_connection(app_ctx.CONFIG.database_path, read_only=True) as con:
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
                FROM {app_ctx.CONFIG.universe_name}.locations
                WHERE {where_clause}
                LIMIT 1
            """
            result = con.execute(query).df()

            if not result.empty:
                return result.iloc[0]['location_id']
    except Exception as e:
        logger.error(f"Error finding location by streets: {e}", exc_info=True)

    return None
