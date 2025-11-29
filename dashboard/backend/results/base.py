"""Base result classes for dashboard query results.

This module provides base classes for wrapping query results with
dashboard-specific rendering functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path
import pandas as pd

from streettransformer import STConfig, EmbeddingDB
from streettransformer.query import DatabaseMixin


class DisplayMixin:
    """Mixin providing display and rendering functionality for Dash UI.

    Provides high-level methods for rendering results in different formats:
    - Accordion view (for web dashboard) - uses template method pattern
    - Table view (for CLI or notebooks)
    - JSON export

    Note: Low-level UI component helpers are in frontend.components.helpers
    """

    def render_table(self, results: pd.DataFrame) -> str:
        """Render results as formatted table.

        Args:
            results: DataFrame to render

        Returns:
            String representation of table
        """
        return results.to_string(index=False)

    def render_json(self, results: pd.DataFrame) -> dict:
        """Render results as JSON.

        Args:
            results: DataFrame to render

        Returns:
            Dictionary with results as list of records
        """
        return results.to_dict(orient='records')

    def get_image_path(self, location_id: int, year: int, extension: str = 'png') -> Path:
        """Get path to image file for a location/year.

        Args:
            location_id: Location ID
            year: Year
            extension: Image file extension

        Returns:
            Path to image file
        """
        universe = self.config.universe_name
        return Path(f"data/{universe}/images/{year}/{location_id}.{extension}")


@dataclass
class BaseQueryResult(ABC, DatabaseMixin, DisplayMixin):
    """Abstract base for all query results.

    Wraps a DataFrame of results with additional functionality for enrichment
    and rendering. Uses template method pattern for rendering - subclasses
    implement _render_accordion_item() to define how individual rows are displayed.

    Attributes:
        results: DataFrame with query results
        config: StreetTransformer configuration
        db: EmbeddingDB instance
        db_connection_func: Optional database connection factory
    """
    results: pd.DataFrame
    config: STConfig
    db: EmbeddingDB
    db_connection_func: Optional[Callable] = None

    @abstractmethod
    def enrich(self):
        """Enrich results with additional data (street names, images, etc.).

        Each subclass implements its own enrichment logic.

        Returns:
            Self for method chaining
        """
        pass

    def get_top_n(self, n: int) -> pd.DataFrame:
        """Get top N results.

        Args:
            n: Number of results to return

        Returns:
            DataFrame with top N results
        """
        return self.results.head(n)

    def filter_by_similarity(self, min_similarity: float) -> pd.DataFrame:
        """Filter results by minimum similarity threshold.

        Args:
            min_similarity: Minimum similarity score (0-1)

        Returns:
            DataFrame with filtered results
        """
        if 'similarity' in self.results.columns:
            return self.results[self.results['similarity'] >= min_similarity]
        return self.results

    # Template method pattern for accordion rendering
    def render_accordion(self):
        """Render all results as accordion.

        Uses template method pattern - calls _render_accordion_item() for each row.

        Returns:
            Dash HTML Div with accordion container
        """
        return self._render_accordion(self.results)

    def _render_accordion(self, results: pd.DataFrame):
        """Render a DataFrame as accordion (internal, testable).

        This method is separated for testability - can be tested with any DataFrame
        without needing to modify self.results.

        Args:
            results: DataFrame to render

        Returns:
            Dash HTML Div with accordion container
        """
        from dash import html

        if results.empty:
            return html.Div("No results to display",
                          style={'fontStyle': 'italic', 'textAlign': 'center',
                                'padding': 20, 'color': '#888'})

        items = []
        for i, row in enumerate(results.itertuples(), 1):
            items.append(self._render_accordion_item(row, index=i))

        return html.Div(items, className='accordion-container')

    @abstractmethod
    def _render_accordion_item(self, row, index: int):
        """Render single result row as accordion item (must override).

        Subclasses implement this to define how individual results are displayed.
        Can use helpers from frontend.components.helpers

        Args:
            row: Single row from DataFrame (namedtuple from itertuples())
            index: 1-based index for display

        Returns:
            Dash HTML Div with accordion item
        """
        pass
