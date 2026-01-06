from abc import ABC
from pathlib import Path
from typing import Optional, List
import logging

import numpy as np
from pydantic import BaseModel
from .. import EmbeddingDB
from .mixins import DatabaseMixin

logger = logging.getLogger(__name__)

class QueryResultInstance(ABC, BaseModel):
    location_id: str
    location_key: str
    similarity: float
    distance: float = None
    title: Optional[str] = None
    description: Optional[str] = None
    street_names: Optional[list[str]] = None

    class Config:
        frozen = False
    
    # TODO: Rename to 'enrich_path' throughout
    # TODO: allow for multiple media_types at once. Add in front-end functionality for this
    def enrich_image_path(self, db_connection, universe_name: str, media_types:str='image'):
        ...

    def enrich_street_names(self, db_connection, universe_name: str, include_borough: bool = True) -> None:
        """Enrich with street names from locations table.

        Args:
            db_connection: Active database connection
            universe_name: Name of universe schema
            include_borough: Whether to include borough in street names
        """
        try:
            query = f"""
                SELECT
                    COALESCE(
                        CASE
                            WHEN additional_streets IS NOT NULL
                            THEN array_to_string(additional_streets, ', ')
                            ELSE NULL
                        END,
                        CONCAT(street1, ' & ', street2)
                    ) as street_name
                FROM {universe_name}.locations
                WHERE location_id = '{self.location_id}'
            """

            result = db_connection.execute(query).df()

            if not result.empty and result.iloc[0]['street_name']:
                # Store as a list for consistency
                self.street_names = [result.iloc[0]['street_name']]

                # Set title from street names if not already set
                if not self.title and self.street_names:
                    self.title = ', '.join([x.title() for x in self.street_names])

        except Exception as e:
            logger.warning(f"Failed to enrich street names for location {self.location_id}: {e}")
class StateResultInstance(QueryResultInstance):
    year: int
    image_path: Optional[Path] = None

    def enrich_image_path(self, db_connection, universe_name: str, media_type: str='image') -> None:
        """Enrich with image path from media_embeddings table.

        Args:
            db_connection: Active database connection
            universe_name: Name of universe schema
        """
        try:
            if isinstance(media_type, str):
                media_type_filter = f"AND media_type='{media_type}'"
            elif isinstance(media_type, list):
                media_type_filter = '\t\n'.join(f"AND media_type='{x}'" for x in media_type)
            else:
                raise TypeError(f'Unknown `media_type`: {media_type}')
            
            query = f"""
                SELECT path
                FROM {universe_name}.media_embeddings
                WHERE location_id = '{self.location_id}'
                    AND year = {self.year}
                    {media_type_filter}
                    AND path IS NOT NULL
                LIMIT 1
            """

            result = db_connection.execute(query).df()

            if not result.empty and result.iloc[0]['path']:
                self.image_path = Path(result.iloc[0]['path'])

        except Exception as e:
            logger.warning(f"Failed to enrich image path for location '{self.location_id}' year {self.year}: {e}")


class ChangeResultInstance(QueryResultInstance):
    year_from: int
    year_to: int
    # delta: Optional[np.ndarray] = None
    start_image_path: Optional[Path] = None
    end_image_path: Optional[Path] = None

    def enrich_image_path(self, db_connection, universe_name: str, media_type: str='image') -> None:
        """Enrich with image paths for both start and end years.

        Args:
            db_connection: Active database connection
            universe_name: Name of universe schema
            media_type: Media type filter (default 'media')
        """
        try:
            if isinstance(media_type, str):
                media_type_filter = f'AND media_type="{media_type}"'
            elif isinstance(media_type, list):
                media_type_filter = '\t\n'.join(f'AND media_type="{x}"' for x in media_type)
            else:
                raise TypeError(f'Unknown `media_type`: {media_type}')

            query = f"""
                SELECT year, path
                FROM {universe_name}.media_embeddings
                WHERE location_id = {self.location_id}
                    AND year IN ({self.year_from}, {self.year_to})
                    {media_type_filter}
                    AND path IS NOT NULL
            """

            results = db_connection.execute(query).df()
            print(results)

            if not results.empty:
                for _, row in results.iterrows():
                    if row['year'] == self.year_from and row['path']:
                        self.start_image_path = Path(row['path'])
                    elif row['year'] == self.year_to and row['path']:
                        self.end_image_path = Path(row['path'])

        except Exception as e:
            logger.warning(f"Failed to enrich image paths for location {self.location_id}: {e}")


