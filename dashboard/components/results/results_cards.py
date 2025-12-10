from pathlib import Path
from dash import html
import dash_mantine_components as dmc
from .results_card_base import BaseResultsCard
from ...utils.display import encode_image_to_base64
from streettransformer.query import StateResultInstance, ChangeResultInstance

class ResultsStateCard(BaseResultsCard):
    def __init__(self, id_prefix: str, rank: int, res: StateResultInstance):
        super().__init__(id_prefix=id_prefix, rank=rank, res=res)
        if not isinstance(res, StateResultInstance):
            raise TypeError(f'{res} not a correctly-formatted StateResultsInstance')

        self.year = res.year
        self.image_path = res.image_path
        
    @property
    def _media_content(self):
        if self.image_path.exists():
            img_base64 = encode_image_to_base64(self.image_path)
            if img_base64:
                return dmc.Image(src=img_base64, radius='md', fit='contain', mt='sm')
            else:
                return dmc.Alert("Image could not be loaded", color='yellow', title="Warning", mt='sm')
        else:
            return dmc.Alert("Image not found", color='yellow', title="Warning", mt='sm')
        
        

class ResultsChangeCard(BaseResultsCard):
    def __init__(self, id_prefix: str, rank: int, res: ChangeResultInstance):
        super().__init__(id_prefix=id_prefix, rank=rank, res=res)
        self.before_year = res.before_year
        self.after_year = res.after_year
        self.before_path = res.before_path # TODO: Probably don't actually need the path once we merge with server version
        self.after_path = res.after_path
    
    @property
    def _media_content(self):
        if self.before_path.exists() and self.after_path.exists():
            before_base64 = encode_image_to_base64(self.before_path)
            after_base64  = encode_image_to_base64(self.after_path)
            if before_base64 and after_base64:
                return html.Div([
                    dmc.Image(src=before_base64, radius='md', fit='contain', mt='sm'),
                    # TODO: Arrow?
                    dmc.Image(src=after_base64, radius='md', fit='contain', mt='sm')
                ])
            else:
                return dmc.Alert("Images could not be loaded", color='yellow', title="Warning", mt='sm')
        else:
            return dmc.Alert("Images not found", color='yellow', title="Warning", mt='sm')