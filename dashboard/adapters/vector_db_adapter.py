"""Thin adapter for converting VectorDB search results to Dashboard types.

This adapter exists temporarily to bridge the gap between VectorDB and Dashboard types.
In the future, when we redesign the system, this adapter can be removed entirely by:
1. Having VectorDB directly return StateResultInstance/ChangeResultInstance
2. Or having the Dashboard consume SearchHit/ChangeVector directly

To remove this adapter in the future:
- Replace calls to `VectorDBAdapter.to_state_results(hits, ...)` with direct construction
- Search for "VectorDBAdapter" in the codebase and update callsites
"""
import logging
from pathlib import Path
from typing import List

from streettransformer.query.queries.results_instances import StateResultInstance, ChangeResultInstance
from streettransformer.query.queries.results_sets import StateResultsSet, ChangeResultsSet

logger = logging.getLogger(__name__)


class VectorDBAdapter:
    """TEMPORARY: Converts VectorDB results to Dashboard result types.

    This is a thin wrapper that will be removed in a future redesign.
    """

    @staticmethod
    def to_state_results(search_hits: List, location_id_field: str = "location_key") -> StateResultsSet:
        """Convert SearchHit list to StateResultsSet.

        Args:
            search_hits: List of SearchHit objects from VectorDB.search_images()
            location_id_field: Which field from SearchHit to use as location_id (default: "location_key")

        Returns:
            StateResultsSet compatible with Dashboard components

        Note:
            Future: Remove this method and have VectorDB return StateResultInstance directly
        """
        results_set = StateResultsSet()

        for hit in search_hits:
            # Map SearchHit fields to StateResultInstance fields
            location_id_value = getattr(hit, location_id_field, hit.location_key)

            result = StateResultInstance(
                location_id=location_id_value,
                location_key=hit.location_key,
                similarity=hit.similarity,
                year=hit.year,
                image_path=Path(hit.image_path) if hit.image_path else None
            )
            results_set.append(result)

        return results_set

    @staticmethod
    def to_change_results(change_vectors: List, location_id_field: str = "location_key") -> ChangeResultsSet:
        """Convert ChangeVector list to ChangeResultsSet.

        Args:
            change_vectors: List of ChangeVector objects from VectorDB.search_changes()
            location_id_field: Which field to use as location_id (default: "location_key")

        Returns:
            ChangeResultsSet compatible with Dashboard components

        Note:
            Future: Remove this method and have VectorDB return ChangeResultInstance directly
        """
        results_set = ChangeResultsSet()

        for vec in change_vectors:
            location_id_value = getattr(vec, location_id_field, vec.location_key)

            result = ChangeResultInstance(
                location_id=location_id_value,
                location_key=vec.location_key,
                similarity=vec.similarity,
                year_a=vec.year_a,
                year_b=vec.year_b
            )
            results_set.append(result)

        return results_set
