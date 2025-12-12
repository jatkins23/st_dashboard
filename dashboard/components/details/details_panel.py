from dash import html
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent
from typing import List

from ..base import BaseComponent
from .base_modality_viewer import ModalityViewerRegistry

import logging
logger = logging.getLogger(__name__)

class DetailsPanel(BaseComponent):
    """Main details panel container with tabbed modality viewers.

    The panel has a stats header (always visible) and dynamically generates
    tabs for each modality viewer from the registry.

    Args:
        id_prefix: Prefix for component IDs
        modality_viewers: List of modality names to include (default: ['images', 'documents', 'projects'])
    """

    def __init__(self, id_prefix: str = 'details', modality_viewers: List[str] = None):
        super().__init__(id_prefix=id_prefix)

        # Use default modalities if not specified
        if modality_viewers is None:
            modality_viewers = ['images', 'documents', 'projects']

        # Build MODALITY_VIEWERS dict from registry based on requested modalities
        self.MODALITY_VIEWERS = {}
        for name in modality_viewers:
            viewer_info = ModalityViewerRegistry.get_viewer(name)
            if viewer_info:
                self.MODALITY_VIEWERS[name] = viewer_info
            else:
                logger.warning(f"Modality viewer '{name}' not found in registry. Skipping.")

    def register_callbacks(self, app):
        """Register callbacks for the panel."""
        from dash import Input, Output, State, html
        from . import DetailsStatsViewer
        from streettransformer.db.database import get_connection
        from ... import state

        # Panel collapse toggle callback
        @app.callback(
            Output('details-collapse', 'opened'),
            Output('details-collapse-btn', 'children'),
            Input('details-collapse-btn', 'n_clicks'),
            State('details-collapse', 'opened'),
            prevent_initial_call=True
        )
        def toggle_details_panel(n_clicks, is_open):
            """Toggle details panel collapse."""
            new_state = not is_open
            icon = html.I(className='fas fa-chevron-up' if new_state else 'fas fa-chevron-down')
            return new_state, icon

        # Dynamically create outputs for stats + all modality viewers
        outputs = [Output('details-stats-content', 'children')]
        outputs.extend([Output(f'details-{modality}-content', 'children')
                       for modality in self.MODALITY_VIEWERS.keys()])
        outputs.append(Output('details-card', 'style'))

        # Unified details update callback - populates all tabs
        @app.callback(
            outputs,
            Input('selected-location-id', 'data'),
            Input('main-map', 'clickData'),
            Input('active-search-tab', 'data'),
            Input('state-query-params', 'data'),
            Input('change-query-params', 'data'),
            prevent_initial_call=False
        )
        def update_details(selected_location_id, click_data, active_tab, state_params, change_params):
            """Update all detail tabs when location is selected."""
            location_id = None

            # Map click takes priority
            if click_data and 'points' in click_data and len(click_data['points']) > 0:
                point = click_data['points'][0]
                if 'customdata' in point:
                    location_id = point['customdata']
            elif selected_location_id:
                location_id = selected_location_id

            # No location selected - return empty values for all outputs
            if not location_id:
                empty_outputs = [html.Div("Select streets or click on the map", className='text-muted fst-italic')]
                empty_outputs.extend([None] * len(self.MODALITY_VIEWERS))
                empty_outputs.append({'display': 'none'})
                return tuple(empty_outputs)

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
                        error_outputs = [dmc.Alert(f"Location {location_id} not found", c='warning')]
                        error_outputs.extend([None] * len(self.MODALITY_VIEWERS))
                        error_outputs.append({'display': 'block'})
                        return tuple(error_outputs)

                    street_name = result.iloc[0]['street_name']

                    # Get images for image viewer
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
                    if not images_df.empty:
                        images_df = images_df.rename(columns={'path': 'image_path'})

                    # Get query year based on active tab
                    query_year = None
                    if active_tab == 'change':
                        query_year = change_params.get('year_from') if change_params else None
                    else:
                        query_year = state_params.get('year') if state_params else None

                    # Create stats viewer (always visible)
                    stats_viewer = DetailsStatsViewer(location_id=location_id, street_name=street_name)
                    result_outputs = [html.Div(stats_viewer.content)]

                    # Dynamically instantiate and render each modality viewer
                    for modality_key, modality_config in self.MODALITY_VIEWERS.items():
                        viewer_class = modality_config['viewer_class']

                        # Instantiate viewer with appropriate args
                        if modality_key == 'images':
                            viewer = viewer_class(images_df=images_df, query_year=query_year)
                        else:
                            viewer = viewer_class(location_id=location_id)

                        result_outputs.append(html.Div(viewer.content))

                    result_outputs.append({'display': 'block'})
                    return tuple(result_outputs)

            except Exception as e:
                logger.error(f"Error getting location details: {e}", exc_info=True)
                error_outputs = [dmc.Alert(f"Error loading location {location_id}", c='error')]
                error_outputs.extend([None] * len(self.MODALITY_VIEWERS))
                error_outputs.append({'display': 'block'})
                return tuple(error_outputs)

    @property
    def content(self) -> list:
        """Return placeholder content."""
        return []

    @property
    def _header(self) -> DashComponent:
        return html.Div([
            dmc.Text("Location Detail", fw=700, size='md'),
            dmc.ActionIcon(
                html.I(className='fas fa-chevron-down'),
                id='details-collapse-btn',
                variant='subtle',
                size='sm'
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '0.5rem'})

    @property
    def _body(self) -> DashComponent:
        # Dynamically generate tab list from MODALITY_VIEWERS
        tabs_list = dmc.TabsList([
            dmc.TabsTab(config['label'], value=key)
            for key, config in self.MODALITY_VIEWERS.items()
        ])

        # Dynamically generate tab panels from MODALITY_VIEWERS
        tabs_panels = [
            dmc.TabsPanel(
                html.Div(id=f'details-{key}-content'),
                value=key
            )
            for key in self.MODALITY_VIEWERS.keys()
        ]

        # Get first modality as default tab
        default_tab = next(iter(self.MODALITY_VIEWERS.keys()))

        return dmc.Collapse([
            html.Div([
                # Stats header (always visible at top)
                html.Div(id='details-stats-content'),

                # Tabbed modality viewers
                dmc.Tabs(
                    [tabs_list, *tabs_panels],
                    value=default_tab,
                    orientation="horizontal",
                    mt="md"
                )
            ], style={'maxHeight': '100%', 'overflowY': 'auto'})
        ], id='details-collapse', opened=True)

    @property
    def layout(self) -> DashComponent:
        """Return the complete panel with card wrapper (for use in layout)."""
        # Return complete card structure with floating style
        # Note: Card visibility is controlled by the details callback via 'details-card' style
        return html.Div([
            dmc.Card([
                self._header,
                self._body
            ], id='details-card', withBorder=True, shadow='sm', p='md',
               style={'display': 'none'})
        ], style={ # TODO: !! Move to an actual style file
            'position': 'absolute',
            'top': '10px',
            'left': '10px',
            'width': '25%',
            'maxHeight': 'calc(80vh - 20px)',
            'overflowY': 'auto',
            'zIndex': 1000
        })
