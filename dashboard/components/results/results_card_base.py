from dash import html
from dash.development.base_component import Component as DashComponent
import dash_bootstrap_components as dbc
from abc import abstractmethod
from ..base import BaseComponent
from streettransformer.query import QueryResultInstance
from logging import getLogger

logger = getLogger(__name__)

class BaseResultsCard(BaseComponent):
    def __init__(self, id_prefix:str, rank:int, res:QueryResultInstance):
        super().__init__(id_prefix=id_prefix)
        self.rank = rank
        
        # Copy properties from result instance
        self.location_id = res.location_id
        self.location_key = res.location_key
        self.similarity = res.similarity
        self.title = res.title
        self.description = res.description
        self.street_names = res.street_names

    def _dispatch_type(self, res): # This is probably unnecessary
        """Ensure that `res` is a correctly formatted object and dispatch"""
        if not isinstance(res, QueryResultInstance):
            raise TypeError(f'{res} not a correctly-formatted StateResultsInstance')
        
    # TODO: this may go into a submodule 
        # Is this actually used anymore?
    def _content_element(self, var:str, label:str):
        elem = html.Div([
            html.Small([
                html.Strong(f'{label}: '),
                html.Span(var)
            ], className='text-muted')
        ], className='mb-2')
        return elem
    
    def register_callbacks(self, app):
        """Register callbacks for the card."""
        pass

    @property
    @abstractmethod
    def _media_content(self) -> DashComponent:
        ...
    
    @property
    def _location_details(self) -> DashComponent:
        return html.Div([
            html.Small([html.Strong("Location ID: "), str(self.location_id)], className='text-muted'),
            html.Br(),
            html.Small([html.Strong("Year: "), str(self.year)], className='text-muted') if self.year else None,
        ], className='mb-2')
    
    @property
    def layout(self) -> dbc.AccordionItem:
        # Create title with badge
        title = html.Div([
            html.Span(f"#{self.rank} - {self.title}", className='me-2'),
            dbc.Badge(f"{self.similarity:.4f}", color='info', pill=True)
        ], className='d-flex align-items-center justify-content-between w-100')

        # Create content
        content = []

        # Location details
        content.append(self._location_details)

        # Image/s
        content.extend(self._media_content)
        
        return dbc.AccordionItem(
            content,
            title=title,
            item_id=self.Id('item')
        )


