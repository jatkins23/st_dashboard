"""Document viewer component for location details."""

from pathlib import Path
from dash import html
import dash_bootstrap_components as dbc
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
        DATA_PATH = Path(str(os.getenv('DATA_PATH')))
        
        for idx, doc_row in enumerate(self.documents_df.itertuples()):
            # Fix file path 
            doc_path = DATA_PATH.parent / doc_row.page_file_path.replace('pages', 'nyc/pages')
            print(doc_path)
            if doc_path.exists():
                print('exists')
                # PDFs: convert first page (page_num=0) to image
                img_base64 = encode_pdf_to_base64(doc_path, page_num=0)
                if img_base64:
                    carousel_items.append({
                        'key': f'page_{idx}',
                        'src': img_base64,
                        'header': f"Page {idx + 1}",
                        'caption': doc_path.name
                    })

        return carousel_items

    @property
    def content(self) -> list:
        """Generate the document section content.

        Returns:
            List of Dash components for the document section
        """
        if self.documents_df is None or self.documents_df.empty:
            return [
                html.H6("Documents", className='fw-bold mt-3'),
                html.Div("No documents found", className='text-muted fst-italic text-center p-3')
            ]

        carousel_items = self._format_carousel_items()

        if not carousel_items:
            return [
                html.H6("Documents", className='fw-bold mt-3'),
                html.Div("No document pages available", className='text-muted fst-italic text-center p-3')
            ]

        return [
            html.H6("Documents:", className='fw-bold mt-3'),
            dbc.Carousel(
                items=carousel_items,
                controls=True,
                indicators=True,
                interval=None,
                className='mb-3'
            )
        ]
