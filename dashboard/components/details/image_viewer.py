"""Image viewer component for location details."""

from pathlib import Path
from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Optional, Tuple, List

from ....utils.display import encode_image_to_base64


class DetailsImageViewer:
    """Viewer component for location image carousel."""

    def __init__(self, images_df: Optional[pd.DataFrame] = None, query_year: Optional[int] = None):
        self.images_df = images_df
        self.query_year = query_year

    def _format_carousel_items(self) -> Tuple[List[dict], int]:
        """Format images into carousel items.

        Returns:
            Tuple of (carousel_items, query_year_index)
        """
        carousel_items = []
        query_year_index = 0

        for idx, img_row in enumerate(self.images_df.itertuples()):
            img_path = Path(img_row.image_path)
            if img_path.exists():
                img_base64 = encode_image_to_base64(img_path)
                if img_base64:
                    year = img_row.year
                    # Check if this is the query year
                    is_query_year = self.query_year and year == self.query_year
                    if is_query_year:
                        query_year_index = len(carousel_items)

                    # Add border style for query year
                    img_style = {
                        'border': '4px solid #ffc107',
                        'borderRadius': '8px',
                        'boxShadow': '0 0 15px rgba(255, 193, 7, 0.6)'
                    } if is_query_year else {}

                    carousel_items.append({
                        'key': str(year),
                        'src': img_base64,
                        'header': f"Year {year}" + (" (Query Year)" if is_query_year else ""),
                        'caption': '',
                        'img_style': img_style
                    })

        return carousel_items, query_year_index

    @property
    def content(self) -> list:
        """Generate the image carousel section content.

        Returns:
            List of Dash components for the image carousel section
        """
        if self.images_df is None or self.images_df.empty:
            return []

        carousel_items, query_year_index = self._format_carousel_items()

        if not carousel_items:
            return []

        return [
            html.H6("Images:", className='mt-3 mb-2'),
            dbc.Carousel(
                items=carousel_items,
                controls=True,
                indicators=True,
                interval=None,
                className='mb-3',
                active_index=query_year_index
            )
        ]
