"""Dissimilarity search form - find when a location changed most."""

from dash import Input, Output, State
from dash.development.base_component import Component as DashComponent
import dash_mantine_components as dmc

from .base_search_form import BaseSearchForm
from .utils import filter_street_options_by_selection
from ... import state as app_state
from streettransformer.query.queries.ask import ImageToImageDissimilarityQuery

import logging
logger = logging.getLogger(__name__)


class DissimilaritySearchForm(BaseSearchForm):
    """Form for longitudinal dissimilarity search.

    Analyzes a single location over time to find when it changed most.
    """

    SEARCH_TYPE = 'dissimilarity'
    TAB_LABEL = 'State Dissimilarity'
    QUERY_CLASS = ImageToImageDissimilarityQuery
    RESULT_TYPE = 'change'

    def __init__(self, available_years: list, all_streets: list, all_boroughs: list = None):
        """Initialize dissimilarity search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            all_boroughs: List of all unique borough names
        """
        super().__init__(
            available_years=available_years,
            all_streets=all_streets,
            all_boroughs=all_boroughs,
            id_prefix='dissimilarity-form',
            title='Longitudinal Dissimilarity Search'
        )

    def _input_selector(self) -> DashComponent:
        """Return street selector (image-based form)."""
        return self._street_selector()

    def _query_inputs(self) -> list:
        """Return dissimilarity-specific inputs."""
        return [
            # Start year
            dmc.GridCol(
                self._year_selector('year-start', 'Start Year'),
                span=2
            ),
            # End year
            dmc.GridCol(
                self._year_selector('year-end', 'End Year'),
                span=2
            ),
            # Comparison strategy
            dmc.GridCol(
                dmc.Stack([
                    dmc.Text("Comparison", size="sm", fw=500),
                    dmc.Select(
                        id=self.Id('comparison-strategy'),
                        data=[
                            {'label': 'Sequential Years', 'value': 'sequential'},
                            {'label': 'All Pairs', 'value': 'all_pairs'},
                            {'label': 'Baseline Comparison', 'value': 'baseline'}
                        ],
                        value='sequential',
                        clearable=False,
                        size="sm"
                    )
                ], gap="xs"),
                span=2
            )
        ]

    def execute_search(self, state, location_id, year_start, year_end,
                       comparison_strategy='sequential', limit=10,
                       media_type='image', use_faiss=True, use_whitening=False,
                       boroughs=None, **kwargs):
        """Execute dissimilarity search.

        Args:
            state: Application state module
            location_id: Location to analyze
            year_start: Beginning of time range
            year_end: End of time range
            comparison_strategy: How to compare years
            limit: Max number of results
            media_type: Type of media to search
            use_faiss: Whether to use FAISS
            use_whitening: Whether to use whitening
            boroughs: Optional borough filter
            **kwargs: Additional parameters

        Returns:
            ChangeResultsSet: Results showing when location changed most
        """
        # Create and execute query
        query = ImageToImageDissimilarityQuery(
            config=state.config,
            db=state.db,
            location_id=location_id,
            year=year_start,  # For StateMixin compatibility
            year_start=int(year_start),
            year_end=int(year_end),
            comparison_strategy=comparison_strategy,
            limit=int(limit),
            use_faiss=use_faiss,
            use_whitening=use_whitening,
            media_types=[media_type] if isinstance(media_type, str) else media_type
        )

        return query.search()

    def register_callbacks(self, app):
        """Register dissimilarity-specific callbacks."""
        # Borough filtering callback (same pattern as other image-based forms)
        @app.callback(
            Output(self.Id('street-selector'), 'data'),
            Input(self.Id('borough-selector'), 'value'),
            State(self.Id('street-selector'), 'data'),
            prevent_initial_call=True
        )
        def filter_streets_by_borough(selected_boroughs, current_data):
            """Filter street options by selected borough."""
            if not selected_boroughs:
                return [{"label": s, "value": s} for s in self.all_streets]

            # Filter streets by borough
            # For now, return all streets (would need borough-street mapping)
            return current_data
