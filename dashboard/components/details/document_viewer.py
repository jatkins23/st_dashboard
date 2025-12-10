"""Document viewer component for location details."""

from pathlib import Path
from dash import html
import dash_mantine_components as dmc
import pandas as pd
from typing import Optional, List
import logging

from streettransformer.db.database import get_connection
from ... import state
from ...utils.display import encode_pdf_to_base64

import os

logger = logging.getLogger(__name__)


class DetailsDocumentViewer:
    """Viewer component for location documents/metadata."""

    def __init__(self, location_id: Optional[int] = None):
        self.location_id = location_id
        self.documents_df = self._load_documents() if location_id else None

    def _load_documents(self) -> pd.DataFrame:
        """Load document page file paths from database.

        Returns:
            DataFrame with page_file_path column
        """
        if not self.location_id:
            return pd.DataFrame()

        try:
            with get_connection(state.CONFIG.database_path, read_only=True) as con:
                query = f"""
                    SELECT page_file_path
                    FROM {state.CONFIG.universe_name}._location_to_document_page
                    WHERE location_id = {self.location_id}
                    ORDER BY page_file_path
                """
                df = con.execute(query).df()
                print('here')
                print(df)
                return df
        except Exception as e:
            logger.warning(f"Failed to load documents for location {self.location_id}: {e}")
            return pd.DataFrame()

    def _format_carousel_items(self) -> List[dict]:
        """Format document pages into carousel items.

        Returns:
            List of carousel items
        """
        carousel_items = []
        # Expand ~ to actual home directory path
        DATA_PATH = Path(str(os.getenv('DATA_PATH'))).expanduser()

        for idx, doc_row in enumerate(self.documents_df.itertuples()):
            # Construct the full file path and expand ~
            doc_path = (DATA_PATH.parent / doc_row.page_file_path.replace('pages', 'nyc/pages')).expanduser()

            logger.debug(f"Checking document: {doc_path}")

            if doc_path.exists():
                # PDFs: convert first page (page_num=0) to image
                img_base64 = encode_pdf_to_base64(doc_path, page_num=0)
                if img_base64:
                    carousel_items.append({
                        'key': f'page_{idx}',
                        'src': img_base64,
                        'header': f"Page {idx + 1}",
                        'caption': doc_path.name
                    })
                    logger.info(f"Successfully added document page {idx + 1}")
                else:
                    logger.warning(f"Failed to encode PDF: {doc_path}")
            else:
                logger.warning(f"Document file does not exist: {doc_path}")

        logger.info(f"Created {len(carousel_items)} carousel items from {len(self.documents_df)} documents")
        return carousel_items

    @property
    def content(self) -> list:
        """Generate the document section content.

        Returns:
            List of Dash components for the document section
        """
        if self.documents_df is None or self.documents_df.empty:
            return [
                dmc.Title("Documents", order=6, fw=700, mt='md'),
                dmc.Text("No documents found", size='sm', c='dimmed', ta='center', p='md')
            ]

        carousel_items = self._format_carousel_items()

        if not carousel_items:
            return [
                dmc.Title("Documents", order=6, fw=700, mt='md'),
                dmc.Text("No document pages available", size='sm', c='dimmed', ta='center', p='md')
            ]

        # Convert carousel items to dmc.Carousel format
        carousel_slides = []
        for item in carousel_items:
            carousel_slides.append(
                dmc.CarouselSlide(
                    html.Div([
                        dmc.Text(item['header'], fw=600, size='sm', mb='xs'),
                        html.Img(src=item['src'], style={'width': '100%', 'height': 'auto'}),
                        dmc.Text(item['caption'], size='xs', c='dimmed', mt='xs')
                    ])
                )
            )

        return [
            dmc.Title("Documents:", order=6, fw=700, mt='md'),
            dmc.Carousel(
                carousel_slides,
                withControls=True,
                withIndicators=True,
                mb='md'
            )
        ]
