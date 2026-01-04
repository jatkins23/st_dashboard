"""Registry for self-registering search forms."""

from typing import Dict, Type, Optional


class SearchFormRegistry:
    """Global registry for all search forms.

    Search forms auto-register via __init_subclass__ pattern.
    """

    _forms: Dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        search_type: str,
        label: str,
        form_class: Type,
        query_class: Type,
        result_type: str
    ):
        """Register a search form.

        Args:
            search_type: Unique ID ('state-similarity', 'dissimilarity', etc.)
            label: Tab label ('State Similarity', 'Change Description', etc.)
            form_class: Search form class
            query_class: Backend query class
            result_type: 'state' or 'change'
        """
        cls._forms[search_type] = {
            'label': label,
            'form_class': form_class,
            'query_class': query_class,
            'result_type': result_type
        }

    @classmethod
    def get_metadata(cls, search_type: str) -> Optional[dict]:
        """Get metadata for a registered form."""
        return cls._forms.get(search_type)

    @classmethod
    def get_all(cls) -> Dict[str, dict]:
        """Get all registered forms."""
        return cls._forms.copy()

    @classmethod
    def get_ordered(cls, order: list[str]) -> list[tuple[str, dict]]:
        """Get forms in specified order."""
        return [(st, cls._forms[st]) for st in order if st in cls._forms]
