# Dashboard Structure

The dashboard has been refactored into a modular frontend/backend structure.

## Current Structure

```
dashboard/
├── config.py                   # ✅ Dashboard configuration (colors, settings)
├── backend/
│   ├── __init__.py            # ✅ Backend exports
│   └── search.py              # ✅ Search logic (location, text, change)
├── utils/
│   ├── __init__.py            # ✅ Utility exports
│   ├── encoding.py            # ✅ CLIP text encoding, image base64
│   └── enrichment.py          # ✅ Result enrichment with street names
└── frontend/                   # ⚠️ TODO: UI components
    ├── tabs/                   # Tab creation functions
    │   ├── location.py
    │   ├── text.py
    │   ├── change.py
    │   └── stats.py
    ├── components/             # Reusable UI components
    │   └── results.py          # Result formatting/display
    └── layout.py               # Main layout

```

## What's Been Extracted

### ✅ Backend (Complete)
- **`backend/search.py`**: All search logic using `streettransformer`
  - `search_by_state()` - State similarity with FAISS/whitening support
  - `search_by_text()` - Text-to-image search with CLIP
  - `search_change_patterns()` - Change detection
  - `get_embedding_stats()` - Statistics

### ✅ Utilities (Complete)
- **`utils/encoding.py`**: CLIP and image encoding
  - `load_clip_for_text()` - Cached CLIP model loading
  - `encode_text_query()` - Text to embedding
  - `encode_image_to_base64()` - Image encoding for display

- **`utils/enrichment.py`**: Result enrichment
  - `enrich_results_with_streets()` - Add street names to results
  - `enrich_change_results_with_images()` - Add image paths for change results

### ✅ Configuration (Complete)
- **`config.py`**: Colors, settings, constants

## Next Steps for Full UI Refactor

The original dashboard (`/Users/jon/code/st_preprocessing/scripts/embedding_dashboard.py`) is 1200 lines.

To complete the modular refactor:

### 1. Frontend Tabs (Extract from original)
```python
# frontend/tabs/location.py
def create_state_search_tab():
    """Location similarity search UI."""
    # Lines 414-488 from original

# frontend/tabs/text.py
def create_text_search_tab():
    """Text-to-image search UI."""
    # Lines 489-555 from original

# frontend/tabs/change.py
def create_change_search_tab():
    """Change detection UI."""
    # Lines 556-612 from original

# frontend/tabs/stats.py
def create_stats_tab():
    """Statistics overview UI."""
    # Lines 614-622 from original
```

### 2. Frontend Components (Extract from original)
```python
# frontend/components/results.py
def format_results_accordion(results, show_years=False):
    """Format search results as accordion."""
    # Lines 624-723 from original

def format_change_results_accordion(results):
    """Format change results with before/after images."""
    # Lines 725-899 from original
```

### 3. Main Layout (Extract from original)
```python
# frontend/layout.py
def create_layout():
    """Create main dashboard layout."""
    # Lines 251-413 from original
```

### 4. Callbacks (Extract from original)
```python
# callbacks.py
def setup_callbacks(app, config, db):
    """Setup all Dash callbacks."""
    # Lines 900-1157 from original
    # Update to use backend.search functions
```

### 5. Main App (Minimal)
```python
# run.py
from dashboard.frontend.layout import create_layout
from dashboard.callbacks import setup_callbacks

def create_app(config):
    app = dash.Dash(__name__)
    app.layout = create_layout()
    setup_callbacks(app, config, db)
    return app
```

## Quick Start (Using Original for Now)

Until full UI refactor is complete, you can use the original dashboard with updated imports:

```python
# Copy original and update imports:
cp /Users/jon/code/st_preprocessing/scripts/embedding_dashboard.py dashboard/app_original.py

# Update imports in app_original.py:
# from st_preprocessing.embeddings import EmbeddingLoader
# from st_preprocessing.db.db import duckdb_connection
# ↓
# from streettransformer import Config, EmbeddingDB
# from streettransformer.database import get_connection
```

## Benefits of This Structure

1. **Backend/Frontend Separation**: Search logic is independent of UI
2. **Testable**: Backend functions can be tested without Dash
3. **Reusable**: Backend can be used from CLI, API, or other UIs
4. **Modular**: Easy to modify individual components
5. **Clean Dependencies**: Uses `streettransformer` package throughout
