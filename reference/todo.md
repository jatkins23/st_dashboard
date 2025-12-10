
# Dashboard
## Features/Functionality
[ ] !! 4 types of search
[X] Street names
[ ] Collapse Panels
[ ] Click on locations
- Text Search
    [ ] Years not needed
    [ ] !! Implement CLIP

## Design Choices
[X] Objectify everything
[ ] Make the logger actually useful to me.

## Backend
[ ] !! Connect the PGvector backend
[ ] Borough integration

## Styling
[ ] Switch to DMC
[ ] Improve the rendering and reactivity (flexibility?) of Search_bar
[ ] Remove SearchCard header?

# Preprocessing
[X] hash-based Unique ID
[ ] Repull



Ok, I want to do that, I also want to move the registering of callbacks and all the functionality that is currently all in app.py into Dashboard.py. Additionally, I want to see if its possible to move the two files from callbacks into their specific classes. At least details.py could be within details right? Or we could separate the function from the callbacks itself

. Finally, I want to update the UI to include more DMC components.


TODO: remove all references to street decomposition in details panel [X]

Is there a way to auto register the callbacks from the object instantiation?

change search form from `change_image_search_form` to `image_change_search_form`

image_change_search_form
image_state_search_form

base_search_form: layout should be abstract