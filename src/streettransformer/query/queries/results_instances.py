from abc import ABC
from pathlib import Path
from typing import Optional, List
import logging

from pydantic import BaseModel
from ... import EmbeddingDB
from ..mixins import DatabaseMixin

logger = logging.getLogger(__name__)

class QueryResultInstance(ABC, BaseModel):
    location_id: int
    location_key: str
    similarity: float
    title: Optional[str] = None
    description: Optional[str] = None
    street_names: Optional[list[str]] = None

    class Config:
        frozen = False

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
                WHERE location_id = {self.location_id}
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


class ChangeResultInstance(QueryResultInstance):
    year_from: int
    year_to: int
    start_image_path: Optional[Path] = None
    end_image_path: Optional[Path] = None

    def enrich_image_paths(self, db_connection, universe_name: str) -> None:
        """Enrich with image paths for both start and end years.

        Args:
            db_connection: Active database connection
            universe_name: Name of universe schema
        """
        try:
            query = f"""
                SELECT year, image_path
                FROM {universe_name}.image_embeddings
                WHERE location_id = {self.location_id}
                    AND year IN ({self.year_from}, {self.year_to})
                    AND image_path IS NOT NULL
            """

            results = db_connection.execute(query).df()

            if not results.empty:
                for _, row in results.iterrows():
                    if row['year'] == self.year_from and row['image_path']:
                        self.start_image_path = Path(row['image_path'])
                    elif row['year'] == self.year_to and row['image_path']:
                        self.end_image_path = Path(row['image_path'])

        except Exception as e:
            logger.warning(f"Failed to enrich image paths for location {self.location_id}: {e}")


