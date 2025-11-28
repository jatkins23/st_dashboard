"""Result classes for dashboard query results.

This package provides result wrappers for query outputs with:
- Data enrichment (street names, images, etc.)
- Dash-specific rendering (accordion views, tables, JSON)

Example:
    >>> from dashboard.backend.results import StateLocationResult
    >>>
    >>> result = StateLocationResult(
    ...     results=df,
    ...     location_id=123,
    ...     year=2020,
    ...     config=config,
    ...     db=db
    ... )
    >>> result.enrich()  # Add street names and images
    >>> html = result.render_accordion()  # Render as Dash UI
"""

from .base import BaseQueryResult, DisplayMixin
from .concrete import StateLocationResult, ChangeLocationResult

__all__ = [
    # Base classes
    'BaseQueryResult',
    'DisplayMixin',

    # Concrete result classes
    'StateLocationResult',
    'ChangeLocationResult',
]
