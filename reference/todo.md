
# Dashboard
## Bugs
[ ] The search functionality kinda breaks after it is called once
[X] Documents actually ashow up

## Features/Functionality
[ ] !! 4 types of search
[X] Street names
[ ] Collapse Panels
[ ] Click on locations
- Text Search
    [X] Years not needed
    [ ] !! Implement CLIP
[ ] Need search to actually show up with the location

## Design Choices
[X] Objectify everything
[ ] Make the logger actually useful to me.

## Backend
[ ] !! Connect the PGvector backend
[ ] Borough integration

## Styling
[ ] Switch to DMC
[ ] Improve the rendering and reactivity (flexibility?) of Search_bar

## Minor
[ ] TODO: Maybe swap type and mode? Or find better names for them
[ ] Rearrange the streets in some logical order besides alphabetical
[X] Remove SearchCard header?
[ ] Remove lazy imports

# Preprocessing
[X] hash-based Unique ID
[ ] Re-pull everything
    [X] Images
    [-] Segmentation
    [ ] Split
    [ ] Re-add to `location_year_files`? Turn this into a view?
[ ] Rebuild the views
[ ] Change to `core2.ddb` from `core.ddb`
[X] Need to convert documxents to images? No, we don't!
[ ] **All paths need to exclude ~** 


Ok, I want to do that, I also want to move the registering of callbacks and all the functionality that is currently all in app.py into Dashboard.py. Additionally, I want to see if its possible to move the two files from callbacks into their specific classes. At least details.py could be within details right? Or we could separate the function from the callbacks itself

. Finally, I want to update the UI to include more DMC components.


TODO: remove all references to street decomposition in details panel [X]

Is there a way to auto register the callbacks from the object instantiation?

change search form from `change_image_search_form` to `image_change_search_form`

image_change_search_form
image_state_search_form

base_search_form: layout should be abstract