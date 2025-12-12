
# Dashboard
## Bugs
[ ] The search functionality kinda breaks after it is called once
[X] Documents actually ashow up

## Features/Functionality
[X]] !! 4 types of search
[X] Street names
[ ] Collapse Panels (Toggle)
    [ ] Collapse sub-panels (Accordion)
[ ] Click on locations
- Text Search
    [X] Years not neededyep
    [ ] !! Implement CLIP
[ ] Need search to actually show up with the location

## Design Choices
[X] Objectify everything
[ ] Make the logger actually useful to me
[ ] Break search functions up into helper functions

## Backend
[ ] !! Connect the PGvector backend
[X] Borough integration
[X] need to dynamically retrieve documents when there are too many

## Styling
[X] Switch to DMC
[ ] Improve the rendering and reactivity (flexibility?) of Search_bar

## Minor
[ ] TODO: Maybe swap type and mode? Or find better names for them
[ ] Rearrange the streets in some logical order besides alphabetical
[X] Remove SearchCard header?
[ ] Remove lazy imports
[ ] DetailViewer overhaul
    [ ] Put Borough somewhere in the Location Detail
    [ ] Better way to display street names
    [ ] Header color, more change.
    [X] Switch to tabs
    [X] Document Cache
    [ ] Split Documents by type

## Adding back in Search Options
[ ] use_whitening, use_faiss added to SearchMethodMixin
    [ ] add to QueryMetaData


## VectorDB Refactor
[ ] Rename media columns for consistency
[ ] EmbeddingDB to just like DuckDB or something
[ ] Integrate with QueryInstance and QueryResult functions
[ ] the location_key, location fiasco
[ ] split query functions into different parts that reference each db? I think this makes sense
[ ] I think configs need to be unified. And I need someway for a query to refer to a PG_Config


# Preprocessing
[X] hash-based Unique ID
[X] Re-pull everything
    [X] Images
    [-] Segmentation
    [X] Split
    [X] Re-add to `location_year_files`? Turn this into a view?
[X] Rebuild the views
[X] Change to `core2.ddb` from `core.ddb`
[X] Need to convert documents to images? No, we don't!
[ ] **All paths need to exclude ~** 


Ok, I want to do that, I also want to move the registering of callbacks and all the functionality that is currently all in app.py into Dashboard.py. Additionally, I want to see if its possible to move the two files from callbacks into their specific classes. At least details.py could be within details right? Or we could separate the function from the callbacks itself

. Finally, I want to update the UI to include more DMC components.


TODO: remove all references to street decomposition in details panel [X]

Is there a way to auto register the callbacks from the object instantiation?

change search form from `change_image_search_form` to `image_change_search_form`

image_change_search_form
image_state_search_form

base_search_form: layout should be abstract