# Dashboard UI Redesign

## Overview

The dashboard has been redesigned with a map-centric interface featuring floating panels for results and location details. The new design is modular and extensible for future enhancements.

## Architecture

### Key Components

1. **Search Bar** ([dashboard/frontend/components/search_bar.py](dashboard/frontend/components/search_bar.py))
   - Unified search component that adapts to the selected tab
   - Supports State, Text, Change, and Stats searches
   - All search fields are rendered in a horizontal layout at the top

2. **Map View** ([dashboard/utils/map_utils.py](dashboard/utils/map_utils.py))
   - Full-page Plotly mapbox background
   - Displays all locations with color coding:
     - **Red pins**: Query/selected location
     - **Blue pins**: Search result locations
     - **Gray pins**: Other locations (dimmed)
   - Clickable pins to view location details

3. **Floating Panels** ([dashboard/frontend/layout.py](dashboard/frontend/layout.py))
   - **Results Panel** (Left side, 400px wide)
     - Displays search results in accordion format
     - Appears when a search is executed
     - Closeable with × button

   - **Detail Panel** (Right side, 450px wide)
     - Shows detailed information about a selected location
     - Displays when:
       - Location ID is entered in search fields
       - A pin is clicked on the map
     - Includes:
       - Location name and information
       - Image gallery with year slider (when available)
       - Statistics section
       - Closeable with × button

4. **Results Components** ([dashboard/frontend/components/results.py](dashboard/frontend/components/results.py))
   - Accordion-style result display
   - Shows similarity scores, images, and metadata
   - Supports both regular and change detection results

5. **Details Components** ([dashboard/frontend/components/details.py](dashboard/frontend/components/details.py))
   - Renders location details with images
   - Year selector for multi-year locations
   - Statistics and metadata display

## Layout Structure

```
┌────────────────────────────────────────────────────────────┐
│  Search Bar (Fixed Top)                                     │
│  [Title] [State|Text|Change|Stats] [Search Fields]         │
└────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐                        ┌──────────────┐  │
│  │  Results     │                        │  Location    │  │
│  │  Panel       │    Full Page Map       │  Details     │  │
│  │  (Left)      │    with Pins           │  (Right)     │  │
│  │              │                        │              │  │
│  │  • Result 1  │    🔴 Query Location   │  Name        │  │
│  │  • Result 2  │    🔵 Results          │  Images      │  │
│  │  • Result 3  │    ⚪ Other Locations  │  Stats       │  │
│  └──────────────┘                        └──────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Running the Redesigned Dashboard

### Using the new redesigned interface:

```bash
python -m dashboard.app_redesign --db /path/to/core.ddb --universe lion --port 8050
```

### Using the original interface:

```bash
python -m dashboard.app --db /path/to/core.ddb --universe lion --port 8050
```

### Command-line arguments:

- `--db`: Path to DuckDB database (required)
- `--universe` or `-u`: Universe name (required)
- `--port` or `-p`: Port to run on (default: 8050)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--debug`: Run in debug mode

## Features

### Tab-based Search Types

1. **State** (Location Similarity Search)
   - Find locations similar to a given location
   - Filter by year and target year
   - Optional FAISS and whitening reranking

2. **Text** (Text-to-Image Search)
   - Find locations matching a text description
   - Optional year filtering
   - Optional FAISS and whitening reranking

3. **Change** (Change Pattern Detection)
   - Find locations with similar change patterns
   - Compare two years for a location

4. **Stats** (Database Statistics)
   - View database statistics
   - Total embeddings, years, dimensions, etc.

### Interactions

1. **Search Flow**:
   - Select tab → Enter search parameters → Click Search
   - Results appear in Results Panel (left)
   - Result locations appear as blue pins on map
   - Query location appears as red pin on map

2. **Location Details Flow**:
   - Enter location ID in search field, OR
   - Click on any pin on the map
   - Detail Panel (right) appears with location information

3. **Panel Management**:
   - Click × button to close panels
   - Panels automatically appear when data is available

## Modularity & Extensibility

The new design is built for easy extension:

### Adding New Search Types

1. Add new tab in [layout.py](dashboard/frontend/layout.py#L200-L205)
2. Add search fields in [search_bar.py](dashboard/frontend/components/search_bar.py)
3. Add callback in [app_redesign.py](dashboard/app_redesign.py)
4. Add backend search function in [backend/search.py](dashboard/backend/search.py)

### Customizing Panels

- Panel styles are defined in [layout.py](dashboard/frontend/layout.py)
- Panel widths, positions, and styling can be adjusted via the `style` dictionaries
- Content rendering is separated into modular components

### Extending Map Functionality

- Map creation is in [map_utils.py](dashboard/utils/map_utils.py)
- Color coding, marker sizes, and hover text can be customized
- Additional map layers can be added via Plotly's mapbox API

### Adding to Detail Panel

The Detail Panel supports additional sections:
- Edit [details.py](dashboard/frontend/components/details.py)
- Add new sections after the image gallery
- Sections are rendered vertically in order

## File Structure

```
dashboard/
├── app.py                     # Original app
├── app_redesign.py            # New redesigned app ⭐
├── config.py                  # Color scheme and settings
├── backend/
│   ├── search.py              # Search functions
│   └── results/               # Result processing
├── frontend/
│   ├── layout.py              # Main layout and styling ⭐
│   ├── components/
│   │   ├── search_bar.py      # Unified search bar ⭐
│   │   ├── results.py         # Results accordion
│   │   └── details.py         # Location details
│   └── tabs/                  # Original tab components
└── utils/
    ├── map_utils.py           # Map creation and utilities ⭐
    └── display.py             # Image encoding utilities
```

Files marked with ⭐ are key to the redesign.

## Future Enhancements

Suggested areas for extension:

1. **Detail Panel**:
   - Add external links (Google Maps, Street View, etc.)
   - Add statistics charts (histograms, time series)
   - Add comparison view for multiple locations

2. **Map Interactions**:
   - Add clustering for dense areas
   - Add heatmap overlay
   - Add drawing tools for area selection

3. **Results Panel**:
   - Add filtering and sorting options
   - Add export functionality
   - Add comparison mode

4. **Search Bar**:
   - Add saved searches
   - Add search history
   - Add advanced filters

## Notes

- All backend search functionality remains unchanged
- The original [app.py](dashboard/app.py) is preserved for backward compatibility
- The redesign uses the same color scheme defined in [config.py](dashboard/config.py)
- Map requires location coordinates from the database
