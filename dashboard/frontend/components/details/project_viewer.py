"""Project viewer component for location details."""

from dash import html
from typing import Optional


class DetailsProjectViewer:
    """Viewer component for project-specific information."""

    def __init__(self, location_id: Optional[int] = None):
        self.location_id = location_id

    @property
    def content(self) -> list:
        """Generate the project section content.

        Returns:
            List of Dash components for the project section
        """
        return [
            html.H6("Projects", className='fw-bold mt-3'),
            html.Div("Coming Soon!", className='text-muted fst-italic text-center p-3')
        ]
