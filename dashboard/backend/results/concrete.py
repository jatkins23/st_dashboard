"""Concrete result classes for different query types.

This module provides result classes for:
- StateLocationResult: Results from state-based location queries
- ChangeLocationResult: Results from change pattern queries
"""

from typing import Optional

from streettransformer.query import StateMixin, ChangeMixin
from .base import BaseQueryResult
from ...utils.display import enrich_results_with_streets


class StateLocationResult(BaseQueryResult, StateMixin):
    """Results from state-based location query.

    Wraps query results with enrichment and rendering for location similarity searches.

    Example:
        >>> result = StateLocationResult(
        ...     results=df,
        ...     location_id=123,
        ...     year=2020,
        ...     config=config,
        ...     db=db
        ... )
        >>> result.enrich()  # Add street names, images
        >>> html = result.render_accordion()  # Render as Dash accordion
    """

    # TODO: this should just take in a location query?
    def __init__(
        self,
        results,
        location_id: int,
        year: int,
        config,
        db,
        target_year: Optional[int] = None,
        db_connection_func = None
    ):
        """Initialize StateLocationResult.

        Args:
            results: DataFrame with query results
            location_id: Location identifier
            year: Year of query location
            config: StreetTransformer configuration
            db: EmbeddingDB instance
            target_year: Optional target year searched
            db_connection_func: Optional database connection factory
        """
        # StateMixin fields
        self.location_id = location_id
        self.year = year
        self.target_year = target_year

        # BaseQueryResult fields
        self.results = results
        self.config = config
        self.db = db
        self.db_connection_func = db_connection_func

    def enrich(self):
        """Enrich with street names and image paths.

        Returns:
            Self for method chaining
        """
        # add street names
        self.results = enrich_results_with_streets(
            self.results,
            self.db_connection_func or self.get_connection,
            self.config.universe_name
        )

        # Add image paths
        # TODO: this can probably be more efficient? Add an `enrich_results_with_images` function?
        self.results['image_path'] = self.results.apply(
            lambda row: str(self.get_image_path(row['location_id'], row['year'])),
            axis=1
        )

        return self

    def _render_accordion_item(self, row, index: int):
        """Render single state result as accordion item.

        Uses frontend helpers for consistent UI components.

        Args:
            row: Result row (namedtuple from DataFrame.itertuples())
            index: 1-based result index

        Returns:
            Dash HTML Div with accordion item
        """
        from dash import html
        from ..frontend.components.helpers import (
            render_accordion_header,
            render_image,
            render_detail_row
        )

        # Extract street info if available
        street_info = None
        if hasattr(row, 'additional_streets') and row.additional_streets:
            # TODO: street info should include street1 and street2. But should probably instead be 
            #       defined in separate logic somewhere
            street_info = row.additional_streets

        # Build header using frontend helper
        header = render_accordion_header(
            index=index,
            location_id=row.location_id,
            similarity=row.similarity,
            street_info=street_info
        )

        # Build content
        content_parts = []

        # Add image
        if hasattr(row, 'image_path'):
            content_parts.append(render_image(row.image_path, max_width=300))

        # Add detail rows
        details = []
        if hasattr(row, 'year'):
            details.append(render_detail_row("Year:", str(row.year)))
        if hasattr(row, 'location_key'):
            details.append(render_detail_row("Location Key:", row.location_key))

        if details:
            content_parts.append(html.Div(details, style={'marginTop': 15}))

        content = html.Div(content_parts, className='accordion-content',
                         style={'display': 'none', 'padding': 15})

        return html.Div([header, content], className='accordion-item')


class ChangeLocationResult(BaseQueryResult, ChangeMixin):
    """Results from change pattern query.

    Wraps query results with enrichment and rendering for change pattern searches.

    Example:
        >>> result = ChangeLocationResult(
        ...     results=df,
        ...     location_id=123,
        ...     start_year=2015,
        ...     end_year=2020,
        ...     config=config,
        ...     db=db
        ... )
        >>> result.enrich()
        >>> html = result.render_accordion()
    """

    def __init__(
        self,
        results,
        location_id: int,
        start_year: int,
        end_year: int,
        config,
        db,
        sequential_only: bool = False,
        db_connection_func = None
    ):
        """Initialize ChangeLocationResult.

        Args:
            results: DataFrame with query results
            location_id: Location identifier
            start_year: Starting year for change detection
            end_year: Ending year for change detection
            config: StreetTransformer configuration
            db: EmbeddingDB instance
            sequential_only: Whether results are from sequential year pairs only
            db_connection_func: Optional database connection factory
        """
        # ChangeMixin fields
        self.location_id = location_id
        self.start_year = start_year
        self.end_year = end_year
        self.sequential_only = sequential_only

        # BaseQueryResult fields
        self.results = results
        self.config = config
        self.db = db
        self.db_connection_func = db_connection_func

    def enrich(self):
        """Enrich with street names and image pairs.

        Returns:
            Self for method chaining
        """
        from ..utils.display import enrich_change_results_with_images

        self.results = enrich_change_results_with_images(
            self.results,
            self.db_connection_func or self.get_connection,
            self.config.universe_name
        )

        return self

    def _render_accordion_item(self, row, index: int):
        """Render single change result as accordion item.

        Shows before/after image pair using frontend helpers.

        Args:
            row: Result row (namedtuple from DataFrame.itertuples())
            index: 1-based result index

        Returns:
            Dash HTML Div with accordion item
        """
        from dash import html
        from ..frontend.components.helpers import (
            render_accordion_header,
            render_image_pair,
            render_detail_row
        )

        # Extract street info if available
        street_info = None
        if hasattr(row, 'additional_streets') and row.additional_streets:
            street_info = row.additional_streets

        # Build header
        header = render_accordion_header(
            index=index,
            location_id=row.location_id,
            similarity=row.similarity,
            street_info=street_info
        )

        # Build content with before/after images
        content_parts = []

        # Add image pair (before/after)
        if hasattr(row, 'image_path_from') and hasattr(row, 'image_path_to'):
            content_parts.append(
                render_image_pair(
                    image_path_from=row.image_path_from,
                    image_path_to=row.image_path_to,
                    year_from=row.year_from,
                    year_to=row.year_to,
                    max_width=200
                )
            )

        # Add detail rows
        details = []
        if hasattr(row, 'location_key'):
            details.append(render_detail_row("Location Key:", row.location_key))
        details.append(
            render_detail_row("Change Period:",
                            f"{row.year_from} → {row.year_to}")
        )

        if details:
            content_parts.append(html.Div(details, style={'marginTop': 15}))

        content = html.Div(content_parts, className='accordion-content',
                         style={'display': 'none', 'padding': 15})

        return html.Div([header, content], className='accordion-item')