"""Document viewer component for location details."""

from dash import html
from typing import Optional


class DetailsDocumentViewer:
    """Viewer component for location documents/metadata."""

    def __init__(self, location_id: Optional[int] = None):
        self.location_id = location_id

    @property
    def content(self) -> list:
        """Generate the document section content.

        Returns:
            List of Dash components for the document section
        """
        return [
            html.H6("Documents", className='fw-bold mt-3'),
            html.Div("Coming Soon!", className='text-muted fst-italic text-center p-3')
        ]
