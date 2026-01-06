"""Base search form with common elements shared across all search types."""

from abc import abstractmethod
from typing import Type
from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent

from ..base import BaseComponent


class BaseSearchForm(BaseComponent):
    """Base search form with auto-registration capability.

    Subclasses must define:
    - SEARCH_TYPE: Unique identifier (e.g., 'state-similarity')
    - TAB_LABEL: Display label (e.g., 'State Similarity')
    - QUERY_CLASS: Backend query class
    - RESULT_TYPE: 'state' or 'change'
    - _query_inputs(): Search-specific UI inputs
    - execute_search(): Search logic
    - register_callbacks(): Form-specific callbacks
    """

    # Subclasses MUST define these for auto-registration
    SEARCH_TYPE: str = None
    TAB_LABEL: str = None
    QUERY_CLASS: Type = None
    RESULT_TYPE: str = None

    def __init_subclass__(cls, **kwargs):
        """Auto-register subclasses when they're defined."""
        super().__init_subclass__(**kwargs)

        # Only register if all required attributes are defined
        if all([cls.SEARCH_TYPE, cls.TAB_LABEL, cls.QUERY_CLASS, cls.RESULT_TYPE]):
            from .registry import SearchFormRegistry

            SearchFormRegistry.register(
                search_type=cls.SEARCH_TYPE,
                label=cls.TAB_LABEL,
                form_class=cls,
                query_class=cls.QUERY_CLASS,
                result_type=cls.RESULT_TYPE
            )

    def __init__(self, available_years: list, all_streets: list,
                 all_boroughs: list = None, id_prefix: str = None, title: str = None):
        """Initialize the base search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            all_boroughs: List of all unique borough names
            id_prefix: Prefix for component IDs (e.g., 'state-search-form')
            title: Display title for the card header
        """
        super().__init__(id_prefix=id_prefix)
        self.title = title
        self.available_years = available_years
        self.all_streets = all_streets or []
        self.all_boroughs = all_boroughs or []

    # ===== COMMON ELEMENTS (shared across all forms) =====

    def _street_selector(self) -> DashComponent:
        """Street multi-select for this search form.

        Note: Each form has its own street selector, but they both update
        the same shared 'selected-location-id' store.
        """
        return dmc.MultiSelect(
            id=self.Id('street-selector'),  # Prefixed - unique per form
            label="Select Streets",
            placeholder="Start typing streets...",
            data=[{"label": s, "value": s} for s in self.all_streets],
            searchable=True,
            nothingFoundMessage="No options",
            clearable=True,
            maxValues=4,
            hidePickedOptions=True,
            maxDropdownHeight=300,
            limit=10,
            value=[],
            styles={
                "dropdown": {"zIndex": 9999}
            }
        )

    def _borough_selector(self) -> DashComponent:
        """Borough selector for filtering by borough."""
        return dmc.MultiSelect(
            id=self.Id('borough-selector'),
            label="Borough (optional)",
            placeholder="Select boroughs...",
            data=[{"label": b, "value": b} for b in self.all_boroughs],
            searchable=False,
            clearable=True,
            size="sm",
            styles={
                "dropdown": {"zIndex": 9999}
            }
        )

    def _text_input(self) -> DashComponent:
        """Text input for text-based search forms."""
        return dmc.TextInput(
            id=self.Id('text-input'),
            placeholder="Enter search text...",
            label="Search Text",
            size="sm"
        )

    def _limit_dropdown(self) -> DashComponent:
        """Result limit dropdown (common to all)."""
        return dmc.Select(
            id=self.Id('limit-dropdown'),
            data=[{'label': str(i), 'value': str(i)} for i in [5, 10, 20, 50]],
            value='10',
            clearable=False,
            size="sm"
        )

    def _media_selector(self) -> DashComponent:
        """Media type dropdown (common to all)."""
        return dmc.Select(
            id=self.Id('media-type-selector'),
            data=[
                {'label': 'Images', 'value': 'image'},
                {'label': 'Masks', 'value': 'mask'},
                {'label': 'Side-by-side', 'value': 'sidebyside'}
            ],
            value='image',
            clearable=False,
            size="sm"
        )

    def _method_selector(self, options: list = None) -> list:
        """FAISS/Whitening checkboxes (common to most).

        Args:
            options: List of options to include. Defaults to ['faiss', 'whitening']

        Returns:
            List of components (label + checkboxes)
        """
        options = options or ['faiss', 'whitening']
        if not options:
            return []

        components = []

        if 'faiss' in options:
            components.append(
                dmc.Switch(
                    id=self.Id('use-faiss-checkbox'),
                    label='FAISS',
                    checked=True,
                    size="sm"
                )
            )
        if 'whitening' in options:
            components.append(
                dmc.Switch(
                    id=self.Id('use-whitening-checkbox'),
                    label='Whitening',
                    checked=False,
                    size="sm"
                )
            )

        return components

    def _search_button(self) -> DashComponent:
        """Search button (common to all)."""
        return dmc.Button(
            'Search',
            id=self.Id('search-btn'),
            color='blue',
            variant='filled',
            size='sm',
            style={'marginTop': '16px'}
        )

    # ===== HELPER: Year selector (used by subclasses) =====

    def _year_selector(self, id_suffix: str, label: str = "Year",
                      include_all: bool = False, placeholder: str = None) -> tuple:
        """Create a year dropdown with label.

        Args:
            id_suffix: Suffix for the dropdown ID
            label: Label text to display
            include_all: Whether to include an "All" option
            placeholder: Placeholder text (defaults to "Select {label.lower()}")

        Returns:
            Tuple of (label_component, dropdown_component) for easy layout
        """
        options = [{'label': str(y), 'value': str(y)} for y in self.available_years]
        if include_all:
            options = [{'label': 'All', 'value': 'all'}] + options

        if placeholder is None:
            placeholder = f'Select {label.lower()}'

        # DMC Select component includes label built-in
        dropdown = dmc.Select(
            id=self.Id(id_suffix),
            label=label,
            data=options,
            placeholder=placeholder,
            clearable=True,
            searchable=True,
            size="sm"
        )

        return dropdown

    # ===== ABSTRACT: Subclass-specific inputs =====

    @abstractmethod
    def _input_selector(self) -> DashComponent:
        """Return the input component for this search form.

        Image-based forms should return self._street_selector()
        Text-based forms should return self._text_input()

        Returns:
            DashComponent: The input component (street selector or text input)
        """
        pass

    @abstractmethod
    def _query_inputs(self) -> list:
        """Return list of dbc.Col elements for query-specific inputs.

        Example for state search:
            return [
                dbc.Col([*self._year_selector('year-selector', 'Year')], width=2),
                dbc.Col([*self._year_selector('target-year-selector', 'Target Year (optional)')], width=2),
            ]

        Example for change search:
            return [
                dbc.Col([*self._year_selector('year-from', 'From Year')], width=2),
                dbc.Col([*self._year_selector('year-to', 'To Year')], width=2),
            ]
        """
        pass

    # ===== LAYOUT =====

    @property
    def layout(self) -> DashComponent:
        """Construct the full form layout using DMC components.

        Note: Each form has its own input selector with a unique ID.
        """
        # Build the search form using DMC Grid
        search_components = [
            # Input selector (street selector or text input - unique per form)
            dmc.GridCol(self._input_selector(), span=4),

            # Borough selector (optional filter)
            dmc.GridCol(self._borough_selector(), span=2),

            # Query-specific inputs (year, text, etc.)
            *self._query_inputs(),

            # Limit dropdown with label
            dmc.GridCol(
                dmc.Stack([
                    dmc.Text("Limit", size="sm", fw=500),
                    self._limit_dropdown()
                ], gap="xs"),
                span=1
            ),

            # Media type with label
            dmc.GridCol(
                dmc.Stack([
                    dmc.Text("Media Type", size="sm", fw=500),
                    self._media_selector()
                ], gap="xs"),
                span=2
            ),

            # Options (FAISS/Whitening switches)
            dmc.GridCol(
                dmc.Stack([
                    dmc.Text("Options", size="sm", fw=500),
                    *self._method_selector()
                ], gap="xs"),
                span=2
            ),

            # Search button
            dmc.GridCol(self._search_button(), span=1),
        ]

        return dmc.Paper([
            dmc.Grid(
                search_components,
                gutter="md",
                align="flex-end"
            )
        ],
        shadow="sm",
        p="md",
        radius="md",
        withBorder=True,
        className="search-form-panel"
        )

    # ===== ABSTRACT METHODS =====

    @abstractmethod
    def execute_search(self, state, **kwargs):
        """Execute the search with the given parameters.

        This method should be implemented by each search form subclass to
        handle its specific search logic.

        Args:
            state: Application state module
            **kwargs: Search-specific parameters (location_id, year, etc.)

        Returns:
            QueryResultsSet: The search results (StateResultsSet, ChangeResultsSet, etc.)
        """
        pass

    @abstractmethod
    def register_callbacks(self, app):
        """Register search-specific callbacks (e.g., street filtering)."""
        pass
