from dash import html
import dash_mantine_components as dmc
from ..base import BaseComponent
from dash.development.base_component import Component as DashComponent

from streettransformer.query import QueryResultsSet
from .results_cards import ResultsStateCard, ResultsChangeCard

import logging
logger = logging.getLogger(__name__)

class ResultsPanel(BaseComponent):
    def __init__(self, id_prefix:str='results', results:QueryResultsSet=None):
        super().__init__(id_prefix=id_prefix)
        self.results = results
    
    def register_callbacks(self, app):
        """Register callbacks for the panel."""
        from dash import Input, Output, State, html

        @app.callback(
            Output('results-collapse', 'opened'),
            Output('results-collapse-btn', 'children'),
            Input('results-collapse-btn', 'n_clicks'),
            State('results-collapse', 'opened'),
            prevent_initial_call=True
        )
        def toggle_results_panel(n_clicks, is_open):
            """Toggle results panel collapse.

            Note: dmc.Collapse uses 'opened' prop in version 2.4.0.
            """
            new_state = not is_open
            icon = html.I(className='fas fa-chevron-up' if new_state else 'fas fa-chevron-down')
            return new_state, icon

    @property
    def content(self) -> list:
        """Return just the content (for use in callbacks)."""
        if self.results is None:
            return []
        elif len(self.results) == 0:
            return [dmc.Text("No results found.", size='sm', c='dimmed', italic=True)]
        else:
            accordion_items = [ResultsStateCard(i, i+1, res)() for i, res in enumerate(self.results)] # TODO: !! Convert this to a dispatcher somehow
            accordion = dmc.Accordion(
                accordion_items,
                chevronPosition='right',
                variant='separated'
            )
            return [
                dmc.Title(f"Top {len(self.results)} similar locations:", order=6, c='green', mb='md'),
                accordion
            ]

    @property
    def layout(self) -> DashComponent:
        """Return the complete panel with card wrapper (for use in layout)."""
        content = self.content

        # Return complete card structure with floating style
        return html.Div([
            dmc.Card([
                # Header with collapse button
                html.Div([
                    dmc.Text("Search Results", fw=700, size='md'),
                    dmc.ActionIcon(
                        html.I(className='fas fa-chevron-down'),
                        id='results-collapse-btn',
                        variant='subtle',
                        size='sm'
                    )
                ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'marginBottom': '0.5rem'}),
                # Collapsible content
                dmc.Collapse([
                    html.Div(
                        content,
                        id='results-content',
                        style={'maxHeight': '100%', 'overflowY': 'auto'}
                    )
                ], id='results-collapse', opened=True)
            ], id='results-card', withBorder=True, shadow='sm', p='md',
               style={'display': 'none' if self.results is None else 'block'})
        ], style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'width': '25%',
            'maxHeight': 'calc(80vh - 20px)',
            'overflowY': 'auto',
            'zIndex': 1000
        }) # TODO: refactor the style to something more elegant