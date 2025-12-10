"""Base search form with common elements shared across all search types."""

from abc import abstractmethod
from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash.development.base_component import Component as DashComponent

from ..base import BaseComponent


class BaseSearchForm(BaseComponent):
    """Base search form with common UI elements.

    Subclasses override:
    - _query_inputs() to provide search-specific inputs (year, text, etc.)
    - execute_search() to implement the search logic
    - register_callbacks() for search-specific callbacks (e.g., street filtering)
    """

    def __init__(self, available_years: list, all_streets: list,
                 id_prefix: str, title: str):
        """Initialize the base search form.

        Args:
            available_years: List of available years for dropdowns
            all_streets: List of all unique street names
            id_prefix: Prefix for component IDs (e.g., 'state-search-form')
            title: Display title for the card header
        """
        super().__init__(id_prefix=id_prefix)
        self.title = title
        self.available_years = available_years
        self.all_streets = all_streets or []

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
                "input": {"color": "black"},
                "dropdown": {"color": "black", "zIndex": 9999}
            }
        )

    def _limit_dropdown(self) -> DashComponent:
        """Result limit dropdown (common to all)."""
        return dcc.Dropdown(
            id=self.Id('limit-dropdown'),
            options=[{'label': str(i), 'value': i} for i in [5, 10, 20, 50]],
            value=10,
            clearable=False,
            style={"color": "black"}
        )

    def _media_selector(self) -> DashComponent:
        """Media type dropdown (common to all)."""
        return dcc.Dropdown(
            id=self.Id('media-type-selector'),
            options=[
                {'label': 'Images', 'value': 'image'},
                {'label': 'Masks', 'value': 'mask'},
                {'label': 'Side-by-side', 'value': 'sidebyside'}
            ],
            value='image',
            clearable=False,
            style={"color": "black", "zIndex": 9999}
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

        components = [dbc.Label("Options", size='sm')]

        if 'faiss' in options:
            components.append(
                dbc.Checklist(
                    id=self.Id('use-faiss-checkbox'),
                    options=[{'label': ' FAISS', 'value': 'faiss'}],
                    value=['faiss'],
                    switch=True
                )
            )
        if 'whitening' in options:
            components.append(
                dbc.Checklist(
                    id=self.Id('use-whitening-checkbox'),
                    options=[{'label': ' Whitening', 'value': 'whitening'}],
                    value=[],
                    switch=True
                )
            )

        return components

    def _search_button(self) -> DashComponent:
        """Search button (common to all)."""
        return dbc.Button(
            'Search',
            id=self.Id('search-btn'),
            color='primary',
            className='mt-4'
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
        options = [{'label': str(y), 'value': y} for y in self.available_years]
        if include_all:
            options = [{'label': 'All', 'value': None}] + options

        if placeholder is None:
            placeholder = f'Select {label.lower()}'

        label_component = dbc.Label(label, size='sm')
        dropdown = dcc.Dropdown(
            id=self.Id(id_suffix),
            options=options,
            placeholder=placeholder,
            style={"color": "black", "zIndex": 9999}
        )

        return label_component, dropdown

    # ===== ABSTRACT: Subclass-specific inputs =====

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
        """Construct the full form layout.

        Note: Each form has its own street selector with a unique ID.
        """
        search_components = [
            # Street selector (unique per form)
            dbc.Col([self._street_selector()], width=3),

            # Query-specific inputs (year, text, etc.)
            *self._query_inputs(),

            # Limit dropdown
            dbc.Col([
                dbc.Label("Limit", size='sm'),
                self._limit_dropdown()
            ], width=1),

            # Media type
            dbc.Col([
                dbc.Label("Media Type", size='sm'),
                self._media_selector()
            ], width=2),

            # Options (FAISS/Whitening)
            dbc.Col(self._method_selector(), width=2),

            # Search button
            dbc.Col([self._search_button()], width=1),
        ]

        return dbc.Card([
            dbc.CardHeader(self.title, className='fw-bold'),
            dbc.CardBody([
                dbc.Row(search_components, className='g-3', align='end')
            ])
        ])

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
