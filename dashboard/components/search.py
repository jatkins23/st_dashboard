"""Search-related helper functions.

Note: Most search functionality has been moved into the individual search form classes.
This module is kept for backward compatibility but may be removed in the future.
"""

import logging

logger = logging.getLogger(__name__)

# All search helper functions have been moved to:
# - get_location_from_streets -> dashboard/components/search_form/utils.py
# - execute_image_search -> StateSearchForm.execute_search()
# - execute_change_search -> ChangeSearchForm.execute_search()
