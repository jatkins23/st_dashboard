"""Project viewer component for location details."""

from dash import html
import dash_bootstrap_components as dbc
from typing import Optional
from .... import state


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
        if not self.location_id or state.PROJECTS_DF is None or state.PROJECTS_DF.empty:
            return [
                html.H6("Projects", className='fw-bold mt-3'),
                html.Div("No projects found", className='text-muted fst-italic text-center p-3')
            ]

        # Filter projects for this location
        location_projects = state.PROJECTS_DF[state.PROJECTS_DF['location_id'] == self.location_id]

        if location_projects.empty:
            return [
                html.H6("Projects", className='fw-bold mt-3'),
                html.Div("No projects at this location", className='text-muted fst-italic text-center p-3')
            ]

        # Create table rows
        table_header = [
            html.Thead(html.Tr([
                html.Th("Project Title"),
                html.Th("Agency"),
                html.Th("Year"),
                html.Th("Status")
            ]))
        ]

        table_rows = []
        for _, row in location_projects.iterrows():
            table_rows.append(html.Tr([
                html.Td(row.get('Project Title', 'N/A'), className='small'),
                html.Td(row.get('Lead Agency', 'N/A'), className='small'),
                html.Td(row.get('Project Year', 'N/A'), className='small'),
                html.Td(row.get('Project Status', 'N/A'), className='small')
            ]))

        table_body = [html.Tbody(table_rows)]

        return [
            html.H6("Projects", className='fw-bold mt-3'),
            dbc.Table(
                table_header + table_body,
                bordered=True,
                hover=True,
                responsive=True,
                size='sm',
                className='mb-3'
            )
        ]
