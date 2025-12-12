"""Base class for modality viewers with self-registration."""

from typing import Dict, Type, Optional


class ModalityViewerRegistry:
    """Registry for all modality viewers."""
    _viewers: Dict[str, dict] = {}

    @classmethod
    def register(cls, name: str, label: str, viewer_class: Type):
        """Register a modality viewer.

        Args:
            name: Unique name for the modality (e.g., 'images', 'documents')
            label: Display label for the tab (e.g., 'Images', 'Documents')
            viewer_class: The viewer class to register
        """
        cls._viewers[name] = {
            'label': label,
            'viewer_class': viewer_class
        }

    @classmethod
    def get_viewer(cls, name: str) -> Optional[dict]:
        """Get a registered viewer by name."""
        return cls._viewers.get(name)

    @classmethod
    def get_all_viewers(cls) -> Dict[str, dict]:
        """Get all registered viewers."""
        return cls._viewers.copy()

    @classmethod
    def get_viewer_names(cls) -> list:
        """Get all registered viewer names."""
        return list(cls._viewers.keys())


class BaseModalityViewer:
    """Base class for modality viewers that auto-registers subclasses.

    Subclasses should define:
        - MODALITY_NAME: str (e.g., 'images', 'documents')
        - MODALITY_LABEL: str (e.g., 'Images', 'Documents')
        - content property: returns list of Dash components
    """

    MODALITY_NAME: str = None
    MODALITY_LABEL: str = None

    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses when they're defined."""
        super().__init_subclass__(**kwargs)

        # Only register if MODALITY_NAME and MODALITY_LABEL are defined
        if cls.MODALITY_NAME and cls.MODALITY_LABEL:
            ModalityViewerRegistry.register(
                name=cls.MODALITY_NAME,
                label=cls.MODALITY_LABEL,
                viewer_class=cls
            )

    @property
    def content(self) -> list:
        """Generate the viewer content.

        Returns:
            List of Dash components
        """
        raise NotImplementedError("Subclasses must implement content property")
