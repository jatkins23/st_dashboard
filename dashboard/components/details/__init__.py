"""Details panel components."""

# Import base viewer registry first
from .base_modality_viewer import BaseModalityViewer, ModalityViewerRegistry

# Import viewers (they will auto-register)
from .stats_viewer import DetailsStatsViewer
from .image_viewer import DetailsImageViewer
from .document_viewer import DetailsDocumentViewer
from .project_viewer import DetailsProjectViewer

# Import panel last (depends on registry)
from .details_panel import DetailsPanel

__all__ = [
    'DetailsPanel',
    'DetailsStatsViewer',
    'DetailsImageViewer',
    'DetailsDocumentViewer',
    'DetailsProjectViewer',
    'BaseModalityViewer',
    'ModalityViewerRegistry',
]