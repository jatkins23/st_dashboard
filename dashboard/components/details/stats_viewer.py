"""Stats viewer component for location details."""

from dash import html
import dash_mantine_components as dmc
from typing import Optional


class DetailsStatsViewer:
    """Viewer component for location stats/header information."""

    def __init__(self, location_id: Optional[int] = None, street_name: Optional[str] = None):
        self.location_id = location_id
        self.street_name = street_name

    @property
    def content(self) -> list:
        """Generate the stats/header section content.

        Returns:
            List of Dash components for the stats section
        """
        if not self.location_id or not self.street_name:
            return []

        return [
            dmc.Title(f"Location {self.location_id}", order=6, fw=700),
            dmc.Text(self.street_name, c='dimmed'),
            dmc.Divider(my='sm')
        ]
