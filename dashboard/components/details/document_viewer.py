"""Document viewer component for location details with caching and pagination."""

from pathlib import Path
from dash import html
import dash_mantine_components as dmc
import pandas as pd
from typing import Optional, List
import logging

from streettransformer.db.database import get_connection
from ...utils.display import encode_pdf_to_base64
from ...utils.document_cache import DocumentImgCache
from .base_modality_viewer import BaseModalityViewer

import os

logger = logging.getLogger(__name__)

# Global cache instance (singleton)
_document_cache = None

def get_document_cache() -> DocumentImgCache:
    """Get or create global document cache instance."""
    global _document_cache
    if _document_cache is None:
        _document_cache = DocumentImgCache()
    return _document_cache


class DetailsDocumentViewer(BaseModalityViewer):
    """Viewer component for location documents with caching and pagination."""

    MODALITY_NAME = 'documents'
    MODALITY_LABEL = 'Documents'

    def __init__(self, location_id: Optional[str] = None, page_size: int = 10):
        """Initialize document viewer.

        Args:
            location_id: Location to view documents for
            page_size: Number of documents to load at once (default: 10)
        """
        self.location_id = location_id
        self.page_size = page_size
        self.cache = get_document_cache()
        self.documents_df = self._load_documents() if location_id else None

    def _load_documents(self) -> pd.DataFrame:
        """Load document page file paths from database.

        Returns:
            DataFrame with page_file_path column
        """
        from ... import context as app_ctx

        if not self.location_id:
            return pd.DataFrame()

        try:
            with get_connection(app_ctx.CONFIG.database_path, read_only=True) as con:
                query = f"""
                    SELECT page_file_path
                    FROM {app_ctx.CONFIG.universe_name}._location_to_document_page
                    WHERE location_id = '{self.location_id}'
                    ORDER BY page_file_path
                """
                df = con.execute(query).df()
                logger.info(f"Loaded {len(df)} document paths for location {self.location_id}")
                return df
        except Exception as e:
            logger.warning(f"Failed to load documents for location '{self.location_id}': {e}")
            return pd.DataFrame()

    def _format_carousel_items(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[dict]:
        """Format document pages into carousel items with pagination.

        Args:
            start_idx: Starting index for pagination
            end_idx: Ending index for pagination (None = load all remaining)

        Returns:
            List of carousel items
        """
        from ... import context as app_ctx

        carousel_items = []
        DATA_PATH = Path(str(os.getenv('DATA_PATH'))).expanduser()

        if end_idx is None:
            end_idx = len(self.documents_df)

        # Limit to page_size documents
        actual_end = min(start_idx + self.page_size, end_idx)
        paginated_docs = self.documents_df.iloc[start_idx:actual_end]

        logger.info(f"Loading documents {start_idx} to {actual_end} (of {len(self.documents_df)} total)")

        for idx, doc_row in enumerate(paginated_docs.itertuples(), start=start_idx):
            doc_path = (DATA_PATH.parent / doc_row.page_file_path.replace('pages', f'{app_ctx.CONFIG.universe_name}/pages')).expanduser()

            if doc_path.exists():
                # Use cache for PDF conversion
                img_base64 = encode_pdf_to_base64(doc_path, page_num=0, cache=self.cache)
                if img_base64:
                    carousel_items.append({
                        'key': f'page_{idx}',
                        'src': img_base64,
                        'header': f"Page {idx + 1} of {len(self.documents_df)}",
                        'caption': doc_path.name
                    })
                else:
                    logger.warning(f"Failed to encode PDF: {doc_path}")
            else:
                logger.warning(f"Document file does not exist: {doc_path}")

        logger.info(f"Created {len(carousel_items)} carousel items (cached conversions used where available)")
        return carousel_items

    @property
    def content(self) -> list:
        """Generate the document section content with loading indicator and pagination.

        Returns:
            List of Dash components for the document section
        """
        if self.documents_df is None or self.documents_df.empty:
            return [
                dmc.Title("Documents", order=6, fw=700, mt='md'),
                dmc.Text("No documents found", size='sm', c='dimmed', ta='center', p='md')
            ]

        total_docs = len(self.documents_df)

        # Show initial batch with pagination info
        return [
            dmc.Title(f"Documents ({total_docs} total):", order=6, fw=700, mt='md'),
            dmc.Stack([
                # Load first batch
                self._create_carousel(start_idx=0),

                # Pagination info if needed
                dmc.Text(
                    f"Showing first {min(self.page_size, total_docs)} of {total_docs} documents",
                    size='xs',
                    c='dimmed',
                    ta='center',
                    mt='xs'
                ) if total_docs > self.page_size else None
            ], gap='xs')
        ]

    def _create_carousel(self, start_idx: int = 0) -> dmc.Carousel:
        """Create carousel component for a batch of documents.

        Args:
            start_idx: Starting index for this batch

        Returns:
            dmc.Carousel component
        """
        carousel_items = self._format_carousel_items(start_idx=start_idx)

        if not carousel_items:
            return dmc.Text("No document pages available", size='sm', c='dimmed', ta='center', p='md')

        # Convert carousel items to dmc.Carousel format
        carousel_slides = []
        for item in carousel_items:
            carousel_slides.append(
                dmc.CarouselSlide(
                    html.Div([
                        dmc.Text(item['header'], fw=600, size='sm', mb='xs'),
                        html.Img(src=item['src'], className='carousel-image'),
                        dmc.Text(item['caption'], size='xs', c='dimmed', mt='xs')
                    ])
                )
            )

        return dmc.Carousel(
            carousel_slides,
            withControls=True,
            withIndicators=True,
            mb='md'
        )
