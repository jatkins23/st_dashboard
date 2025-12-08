from dash import html
import dash_bootstrap_components as dbc
from ..base import BaseComponent
from ....utils.display import encode_image_to_base64

from streettransformer.query import StateResultInstance, ChangeResultInstance

class ResultsStateCard(BaseComponent):
    def __init__(self, id_prefix:str, rank:int, res:StateResultInstance):
        super().__init__(id_prefix=id_prefix)
        self.rank = rank

        if not isinstance(res, StateResultInstance):
            raise ValueError(f'{res} not a correctly-formatted StateResultsInstance')

        # Copy properties from result instance
        self.location_id = res.location_id
        self.location_key = res.location_key
        self.similarity = res.similarity
        self.title = res.title
        self.description = res.description
        self.street_names = res.street_names
        self.year = res.year
        self.image_path = res.image_path

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
    def layout(self) -> dbc.AccordionItem:
        from pathlib import Path

        # Create title with badge
        title = html.Div([
            html.Span(f"#{self.rank} - {self.title}", className='me-2'),
            dbc.Badge(f"{self.similarity:.4f}", color='info', pill=True)
        ], className='d-flex align-items-center justify-content-between w-100')

        # Create content
        content = []

        # Location details
        content.append(
            html.Div([
                html.Small([html.Strong("Location ID: "), str(self.location_id)], className='text-muted'),
                html.Br(),
                html.Small([html.Strong("Year: "), str(self.year)], className='text-muted') if self.year else None,
            ], className='mb-2')
        )

        # Location key if different from title
        if self.location_key and self.location_key != self.title:
            content.append(self._content_element(self.location_key, 'Location Key'))
            # content.append(
            #     html.Small([html.Strong("Location Key: "), self.location_key], className='text-muted d-block mb-2')
            # )

        # Image if provided
        if self.image_path:
            img_path = Path(self.image_path) if not isinstance(self.image_path, Path) else self.image_path
            if img_path.exists():
                img_base64 = encode_image_to_base64(img_path)
                if img_base64:
                    content.append(
                        html.Img(src=img_base64, className='img-fluid rounded mt-2')
                    )
                else:
                    content.append(
                        dbc.Alert("Image could not be loaded", color='warning', className='mt-2 small')
                    )
            else:
                content.append(
                    dbc.Alert("Image not found", color='warning', className='mt-2 small')
                )

        return dbc.AccordionItem(
            content,
            title=title,
            item_id=self.Id('item')
        )


