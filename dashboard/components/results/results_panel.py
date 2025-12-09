from dash import html
import dash_bootstrap_components as dbc
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
            Output('results-collapse', 'is_open'),
            Output('results-collapse-btn', 'children'),
            Input('results-collapse-btn', 'n_clicks'),
            State('results-collapse', 'is_open'),
            prevent_initial_call=True
        )
        def toggle_results_panel(n_clicks, is_open):
            """Toggle results panel collapse."""
            new_state = not is_open
            icon = html.I(className='fas fa-chevron-up' if new_state else 'fas fa-chevron-down')
            return new_state, icon

    @property
    def content(self) -> list:
        """Return just the content (for use in callbacks)."""
        if self.results is None:
            return []
        elif len(self.results) == 0:
            return [html.Div("No results found.", className='text-muted fst-italic')]
        else:
            accordion_items = [ResultsStateCard(i, i+1, res)() for i, res in enumerate(self.results)] # TODO: !! Convert this to a dispatcher somehow
            accordion = dbc.Accordion(
                accordion_items,
                start_collapsed=True,
                always_open=False,
                flush=True
            )
            return [
                html.H6(f"Top {len(self.results)} similar locations:", className='text-success mb-3'),
                accordion
            ]

    @property
    def layout(self) -> DashComponent:
        """Return the complete panel with card wrapper (for use in layout)."""
        content = self.content

        # Return complete card structure with floating style
        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("Search Results", className='fw-bold'),
                        dbc.Button(
                            html.I(className='fas fa-chevron-down'),
                            id='results-collapse-btn',
                            color='link', size='sm', className='ms-auto p-0'
                        )
                    ], className='d-flex align-items-center justify-content-between')
                ]),
                dbc.Collapse([
                    dbc.CardBody(
                        content,
                        id='results-content',
                        style={'maxHeight': '100%', 'overflowY': 'auto'}
                    )
                ], id='results-collapse', is_open=True)
            ], id='results-card', style={'display': 'none' if self.results is None else 'block'})
        ], style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'width': '25%',
            'maxHeight': 'calc(80vh - 20px)',
            'overflowY': 'auto',
            'zIndex': 1000
        }) # TODO: refactor the style to something more elegant