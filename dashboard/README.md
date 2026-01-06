# Dashboard Architecture

The `StreetTransformer` dashboard presents a user interface to allow users to query the `StreetTransformer` database.
- The data for this application comes from the `st_preprocessing` package
- The search functionality is contained in this repo within src/ (separate ReadMe there)

Architecturally, the dashboard is a class-based Dash application that treats each visual component as an encapsulated object and related functionality as an **component registry pattern** for extensibility.

## Design Philosophy

- **Class-based components**: All UI elements are classes inheriting from `BaseComponent`, which contain their own callback definitions for backend interaction
- **Registry Patterns**: For each component that duplicates itself for multiple __s (e.g. Search types or Modality Viewers), a registry pattern is used to keep track of each flavor of the component. Those 
- **Modular composition**: Components are composed of smaller sub-components with a goal of making code readable and logically composed. Note: refactoring is needed for the new master tab structure
- **Separation**: UI components and UI functionality are defined here. Whereas the queries that actually hit the database should be contained within the `streettransformer` library)

## Search Functionality Overview
Search is handled between the frontend `dashboard`

## Directory Structure

```
dashboard/
├── app.py                          # Main Dash application entry point
├── config.py                       # Dashboard configuration (colors, settings)
├── context.py                      # Global application context (e.g. database path, etc. defined at runtime)
│
├── components/                     # UI components
│   ├── base.py                     # BaseComponent with ID management
│   ├── dashboard.py                # Main Dashboard orchestrator (composes the 4 sub-modules into a single dashboard)
│   ├── helpers.py                  # Legacy helper utilities
│   │
│   ├── search_form/                # The main user-interactive interface. One flavor per query type
│   │   ├── __init__.py             # Imports to trigger registration
│   │   ├── base_search_form.py     # BaseSearchForm + auto-registration
│   │   ├── registry.py             # SearchFormRegistry
│   │   ├── utils.py                # Form utilities
│   │   ├── state_similarity_search_form.py
│   │   ├── state_description_search_form.py
│   │   ├── state_dissimilarity_search_form.py
│   │   ├── change_similarity_search_form.py
│   │   └── change_description_search_form.py
│   │
│   |── map/                        # Primary visual component of the search tab. a map that displays projects, results, and controls the query inputs via clicking
│   |   └── map_utils.py            
│   │
│   ├── details/                    # Popover panel on left that shows details of the selected location
│   │   ├── __init__.py             # Imports to trigger registration
│   │   ├── base_modality_viewer.py # BaseModalityViewer + ModalityViewerRegistry
│   │   ├── details_panel.py        # DetailsPanel orchestrator
│   │   ├── details_utils.py        # Shared utilities
│   │   ├── image_viewer.py         # DetailsImageViewer
│   │   ├── document_viewer.py      # DetailsDocumentViewer
│   │   ├── project_viewer.py       # DetailsProjectViewer
│   │   └── stats_viewer.py         # DetailsStatsViewer
│   │
│   └── results/                    # Popover panel on the right that shows results of the queries. Interacts with class instances from streettransformer/queries
│       ├── __init__.py
│       ├── results_panel.py        # ResultsPanel orchestrator
│       ├── results_card_base.py    # BaseResultCard
│       └── results_cards.py        # Concrete card implementations
│
└── utils/                          # Dashboard utilities
    ├── __init__.py
    ├── encoding.py                 # CLIP encoding, image base64
    ├── display.py                  # Display utilities
    └── document_cache.py           # PDF document caching
```

## Component Architecture

### Base Classes

#### BaseComponent

All dashboard components inherit from `BaseComponent` which provides ID management:

```python
from dashboard.components.base import BaseComponent

class SampleComponent(BaseComponent):
    def __init__(self, id_prefix="my-component"):
        super().__init__(id_prefix=id_prefix)

    def Id(self, suffix):
        """Generate unique component ID"""
        return self.get_id(suffix)

    @property
    def layout(self):
        return dmc.Paper(id=self.Id('container'), children=[...])
```

**Benefits:**
- Consistent ID generation: `my-component-container`, `my-component-button`, etc.
- Avoids ID collisions between component instances
- Clean callback registration with unique IDs

### Registry Pattern

#### Search Forms (Self-Registering)

Search forms automatically register themselves when imported:

```python
# In search_form/state_similarity_search_form.py
from .base_search_form import BaseSearchForm
from streettransformer.query.queries import StateSimilarityQuery

class StateSimilaritySearchForm(BaseSearchForm):
    # Define class attributes for registration
    SEARCH_TYPE = 'state-similarity'
    TAB_LABEL = 'State Similarity'
    QUERY_CLASS = StateSimilarityQuery
    RESULT_TYPE = 'state'

    def _input_selector(self):
        """Return street selector for image-based queries"""
        return self._street_selector()

    def _query_inputs(self):
        """Return form-specific inputs (year selectors, etc.)"""
        return [
            dmc.GridCol(self._year_selector('year', 'Year'), span=2),
            dmc.GridCol(self._year_selector('target-year', 'Target Year'), span=2),
        ]

    def execute_search(self, state, **kwargs):
        """Execute search using StateSimilarityQuery"""
        # Implementation here
        pass

    def register_callbacks(self, app):
        """Register form-specific callbacks"""
        # Implementation here
        pass
```

**How it works:**
1. `BaseSearchForm.__init_subclass__()` automatically registers the class
2. `SearchFormRegistry` stores: `{search_type: {label, form_class, query_class, result_type}}`
3. Main dashboard retrieves forms from registry in desired order
4. No manual registration needed!

**Available search forms:**
- `state-similarity`: Image-to-Image similarity at a point in time
- `state-description`: Text-to-Image search at a point in time
- `state-dissimilarity`: Image-to-Image dissimilarity
- `change-similarity`: Image Pair-to-Image Pair similarity across time
- `change-description`: Text-to-Image Pair search

#### Modality Viewers (Self-Registering)

Modality viewers auto-register for the details panel:

```python
# In details/image_viewer.py
from .base_modality_viewer import BaseModalityViewer
from dash import html

class DetailsImageViewer(BaseModalityViewer):
    # Define for registration
    MODALITY_NAME = 'images'
    MODALITY_LABEL = 'Images'

    def __init__(self, location_id=None):
        self.location_id = location_id

    @property
    def content(self):
        """Generate viewer content (list of Dash components)"""
        return [
            html.Div(f"Images for location {self.location_id}")
        ]
```

**How it works:**
1. `BaseModalityViewer.__init_subclass__()` registers the class
2. `ModalityViewerRegistry` stores: `{modality_name: {label, viewer_class}}`
3. `DetailsPanel` instantiates viewers from registry based on configuration
4. Flexible ordering: `DetailsPanel(modality_viewers=['images', 'documents', 'projects'])`

**Available modality viewers:**
- `images`: Street imagery viewer
- `documents`: PDF document viewer
- `projects`: Related projects viewer
- `stats`: Location statistics viewer

### Component Hierarchy

```
Dashboard (Main Orchestrator)
└─── Search Master Panel (Now one of five)
    ├── Search Form Tabs (from SearchFormRegistry)
    │   ├── StateSimilaritySearchForm
    │   ├── StateDescriptionSearchForm
    │   ├── StateDissimilaritySearchForm
    │   ├── ChangeSimilaritySearchForm
    │   └── ChangeDescriptionSearchForm
    |
    ├── MapPanel
    │
    ├── ResultsPanel
    │   ├── StateResultCard (for state queries)
    │   └── ChangeResultCard (for change queries)
    │
    └── DetailsPanel
        └── Modality Tabs (from ModalityViewerRegistry)
            ├── DetailsImageViewer
            ├── DetailsDocumentViewer
            ├── DetailsProjectViewer
            └── DetailsStatsViewer
```

## Data Flow

### Search Flow

1. **User interacts with search form** selects location, year, etc necessary for query
2. **DetailsPanel updates** Renders available data for each modality (see below)
3. **Form callback triggers** with user inputs (location, year, options)
4. **execute_search() called** on the form instance
5. **Query class instantiated** (e.g., StateSimilarityQuery from `streettransformer`)
6. **Query executes** against database thru `_execute_search`
7. **Results returned** as `StateResultsSet` or `ChangeResultsSet` or maybe `DissimilaritySet`
8. **ResultsPanel renders results** using appropriate card type
9. **MapsPanel renders results** Maps panel displays the results

### Details Flow

1. **User clicks location** in results
2. **Callback updates** `selected-location-id` store
3. **DetailsPanel callback triggers** with new location ID
4. **Each viewer refreshes** its content based on new location
5. **Tabs display** updated information

## Configuration

### Dashboard Config

`dashboard/config.py` contains:
- Color schemes for map markers
- Dashboard settings
- Constants and defaults

### Application Context

`dashboard/context.py` provides global state:
TODO: This needs some updating to consolidate databases
```python
from dashboard.context import DashContext

context = DashContext(
    config=STConfig(universe_name="lion"),
    db=EmbeddingDB(config)
)

# Access throughout app
config = context.config
db = context.db
```

## Best Practices

1. **Import at top of files** for clarity (even if lazy loading would work)
2. **Use Pydantic models** for data validation
3. **Keep components focused**: One responsibility per class
4. **Leverage mixins**: Compose functionality from `BaseSearchForm` helpers
5. **Document class attributes**: Especially for registry pattern (SEARCH_TYPE, MODALITY_NAME, etc.)
6. **Test components independently**: Don't require full app context for unit tests


## Migration from Legacy Code -- 

To update legacy code, most 

1. **Identify search logic** → Move to `streettransformer` query classes
2. **Identify UI forms** → Convert to search form classes inheriting from `BaseSearchForm`
3. **Identify result display** → Convert to result card classes
4. **Identify detail views** → Convert to modality viewer classes
5. **Register components** → Import in `__init__.py` files
6. **Update callbacks** → Use component `Id()` methods for consistent IDs

## Extending the Dashboard - This is mostly AI-generated. Not sure how helpful it is for anything

### Adding a New Search Form

1. **Create new file** in `dashboard/components/search_form/`:

```python
# my_custom_search_form.py
from .base_search_form import BaseSearchForm
from streettransformer import MyCustomQuery

class MyCustomSearchForm(BaseSearchForm):
    SEARCH_TYPE = 'my-custom'
    TAB_LABEL = 'My Custom Search'
    QUERY_CLASS = MyCustomQuery
    RESULT_TYPE = 'state'  # or 'change'

    def _input_selector(self):
        return self._street_selector()  # or self._text_input()

    def _query_inputs(self):
        return [
            dmc.GridCol(self._year_selector('year', 'Year'), span=2)
        ]

    def execute_search(self, state, **kwargs):
        # Create and execute query
        query = self.QUERY_CLASS(
            location_id=kwargs['location_id'],
            year=kwargs['year'],
            config=state.config,
            db=state.db,
            **kwargs
        )
        return query.search()

    def register_callbacks(self, app):
        # Optional: register any form-specific callbacks
        pass
```

2. **Import in `__init__.py`**:
```python
from .my_custom_search_form import MyCustomSearchForm
```

3. **Auto-registered!** Form will appear in dashboard tabs

### Adding a New Modality Viewer

1. **Create new file** in `dashboard/components/details/`:

```python
# my_custom_viewer.py
from .base_modality_viewer import BaseModalityViewer
from dash import html

class DetailsMyCustomViewer(BaseModalityViewer):
    MODALITY_NAME = 'my_custom'
    MODALITY_LABEL = 'My Custom Data'

    def __init__(self, location_id=None):
        self.location_id = location_id

    @property
    def content(self):
        if not self.location_id:
            return [html.Div("No location selected")]

        # Fetch and display custom data
        return [
            html.H4(f"Custom data for {self.location_id}"),
            # Add your components here
        ]
```

2. **Import in `__init__.py`**:
```python
from .my_custom_viewer import DetailsMyCustomViewer
```

3. **Configure DetailsPanel** to include it:
```python
DetailsPanel(modality_viewers=['images', 'my_custom', 'documents'])
```

### Adding a New Result Card Type

If you have a new result type (beyond 'state' and 'change'):

1. **Create card class** in `dashboard/components/results/results_cards.py`:

```python
from .results_card_base import BaseResultCard

class MyCustomResultCard(BaseResultCard):
    def __init__(self, result, index, card_id):
        super().__init__(result, index, card_id)

    @property
    def content(self):
        # Customize card layout
        return [...]
```

2. **Update ResultsPanel** to use new card type

## Callbacks

Callbacks are registered in individual components:

- **Search forms**: Register their own search callbacks in `register_callbacks()`
- **DetailsPanel**: Registers callbacks for location selection updates
- **ResultsPanel**: Registers callbacks for result card interactions

## Utilities

### Image Encoding

```python
from dashboard.utils.encoding import encode_image_to_base64

# Convert image path to base64 for display
base64_str = encode_image_to_base64("/path/to/image.png")
```

### Document Caching

```python
from dashboard.utils.document_cache import DocumentCache

cache = DocumentCache(cache_dir=".doc_cache")
pdf_images = cache.get_pdf_images("/path/to/doc.pdf")
```

### Map Generation

```python
from dashboard.components.map.map_utils import create_results_map

folium_map = create_results_map(
    results_df=results.df,
    query_location_id="123",
    color_by_year=True
)
```

## Running the Dashboard

```python
# dashboard/app.py
from dashboard.components.dashboard import Dashboard
from dashboard.context import DashContext
from streettransformer import STConfig, EmbeddingDB

# Setup
config = STConfig(universe_name="lion")
db = EmbeddingDB(config)
context = DashContext(config=config, db=db)

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create dashboard
dashboard = Dashboard(
    config=config,
    db=db,
    available_years=[2015, 2016, 2017, 2018, 2019, 2020, 2021],
    all_streets=["Main St", "Broadway", ...],
    search_form_order=['state-similarity', 'state-description', 'change-similarity']
)

# Set layout and register callbacks
app.layout = dashboard.layout
dashboard.register_callbacks(app)

# Run
if __name__ == '__main__':
    app.run_server(debug=True)
```