from dash import html
import dash_bootstrap_components as dbc
from ..base import BaseComponent
from dash.development.base_component import Component as DashComponent

from streettransformer.query import QueryResultsSet
from .results_state_card import ResultsStateCard
#from .results_change_card import ResultsChangeCard

import logging
logger = logging.getLogger(__name__)

class ResultsPanel(BaseComponent):
    def __init__(self, id_prefix:str='results', results:QueryResultsSet=None):
        super().__init__(id_prefix=id_prefix)
        self.results = results
    
    def register_callbacks(self, app):
        """Register callbacks for the panel."""
        pass

    @property
    def content(self) -> list:
        """Return just the content (for use in callbacks)."""
        if self.results is None:
            return []
        elif len(self.results) == 0:
            return [html.Div("No results found.", className='text-muted fst-italic')]
        else:
            accordion_items = [ResultsStateCard(i, i, res)() for i, res in enumerate(self.results)]
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
                dbc.CardHeader("Search Results", className='fw-bold'),
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
        })