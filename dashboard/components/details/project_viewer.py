"""Project viewer component for location details."""

from dash import html
import dash_mantine_components as dmc
from typing import Optional
from ... import state


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
                dmc.Title("Projects", order=6, fw=700, mt='md'),
                dmc.Text("No projects found", size='sm', c='dimmed', ta='center', p='md')
            ]

        # Filter projects for this location
        location_projects = state.PROJECTS_DF[state.PROJECTS_DF['location_id'] == self.location_id]

        if location_projects.empty:
            return [
                dmc.Title("Projects", order=6, fw=700, mt='md'),
                dmc.Text("No projects at this location", size='sm', c='dimmed', ta='center', p='md')
            ]

        # Create table rows
        table_rows = []
        for _, row in location_projects.iterrows():
            table_rows.append(
                dmc.TableTr([
                    dmc.TableTd(row.get('Project Title', 'N/A')),
                    dmc.TableTd(row.get('Lead Agency', 'N/A')),
                    dmc.TableTd(row.get('Project Year', 'N/A')),
                    dmc.TableTd(row.get('Project Status', 'N/A'))
                ])
            )

        return [
            dmc.Title("Projects", order=6, fw=700, mt='md'),
            dmc.Table([
                dmc.TableThead(
                    dmc.TableTr([
                        dmc.TableTh("Project Title"),
                        dmc.TableTh("Agency"),
                        dmc.TableTh("Year"),
                        dmc.TableTh("Status")
                    ])
                ),
                dmc.TableTbody(table_rows)
            ], striped=True, highlightOnHover=True, withTableBorder=True, mb='md')
        ]
