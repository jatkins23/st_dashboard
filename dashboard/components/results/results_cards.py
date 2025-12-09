from pathlib import Path
from dash import html
import dash_bootstrap_components as dbc
from .results_card_base import BaseResultsCard
from ....utils.display import encode_image_to_base64
from streettransformer.query import StateResultInstance, ChangeResultInstance

class ResultsStateCard(BaseResultsCard):
    def __init__(self, id_prefix:str, res: StateResultInstance):
        super().__init__(id_prefix=id_prefix)
        if not isinstance(res, StateResultInstance):
            raise TypeError(f'{res} not a correctly-formatted StateResultsInstance')
        
        self.year = res.year
        self.media_path = res.media_path
        
    @property
    def _media_content(self):
        if self.media_path.exists():
            # img_path = Path(self._media_path)
            img_base64 = encode_image_to_base64(self.media_path)
            if img_base64:
                return html.Img(src=img_base64, className='img-fluid rounded mt-2')
            else:
                return dbc.Alert("Image could not be loaded", color='warning', className='mt-2 small')
        else:
            return dbc.Alert("Image not found", color='warning', className='mt-2 small')

class ResultsChangeCard(BaseResultsCard):
    def __init__(self, id_prefix:str, res: ChangeResultInstance):
        super().__init__(id_prefix=id_prefix)
        self.before_year = res.before_year
        self.after_year = res.after_year 
        self.before_path = res.before_path # TODO: Probably don't actually need the path once we merge with server version
        self.after_path = res.after_path
    
    @property
    def _media_content(self):
        if self.before_path.exists() and self.after_path.exists(0):
            # img_path = Path(self._media_path)
            before_base64 = encode_image_to_base64(self.media_path)
            after_base64  = encode_image_to_base64(self.media_path)
            if before_base64 and after_base64:
                return html.Span([
                    html.Img(src=before_base64, className='img-fluid rounded mt-2'),
                    # TODO: Arrow?
                    html.Img(src=after_base64, className='img-fluid rounded mt-2')
                ])
            else:
                return dbc.Alert("Images could not be loaded", color='warning', className='mt-2 small')
        else:
            return dbc.Alert("Images not found", color='warning', className='mt-2 small')