import logging
import time

from ..base_query import BaseQuery
from ..mixins import TextQueryMixin, ChangeMixin
from ..results_sets import ChangeResultsSet
from ..results_instances import ChangeResultInstance
from ..metadata import QueryMetadata

from ... import WhiteningTransform


class ChangeDescriptionQuery(TextQueryMixin, ChangeMixin, BaseQuery):
    """Text-to-image change detection query.

    TODO: write a human description.
    Find image pairs that changed in a way similar to a text description.

    Example:
        "buildings being demolished" → pairs showing demolition
        "new construction" → pairs showing new buildings appearing

    Attributes:
        text_query: Text description of desired change
        year_from: Beginning year of change
        year_to: Ending year of change
        sequential_only: Whether to only search sequential year pairs
        clip_encoder: CLIP encoder for text embedding
        target_years: Optional list of years to search within
    """

    def search(self) -> ChangeResultsSet:
        """Execute text-to-image change search."""
        start_time = time.perf_counter()
        logger = logging.getLogger(__name__)

        # 1. Encode text query using CLIP
        if self.clip_encoder is None:
            raise ValueError(
                "clip_encoder is required for text queries. "
                "Create a CLIPEncoder instance and pass it to the query."
            )

        logger.info(f"Encoding text query for change detection: '{self.text_query}'")
        text_embedding = self.clip_encoder.encode(self.text_query)

        # 2. Search change embeddings using vector similarity
        # This searches for changes that match the text description
        results = self.db.search_change_vectors(
            query_delta=text_embedding,
            limit=self.limit,
            year_from=self.year_from,
            year_to=self.year_to
        )

        # 3. Apply whitening reranking if requested
        if self.use_whitening and not results.empty:
            whiten = WhiteningTransform(self.config)
            results = whiten.rerank_results(
                query_vector=text_embedding,
                results=results,
                year=None,
                top_k=self.limit
            )

        # 4. Convert to ChangeResultsSet
        results_dict = results.to_dict(orient='records')
        result_instances = [ChangeResultInstance.model_validate(row) for row in results_dict]

        elapsed_time = (time.perf_counter() - start_time) * 1000

        metadata = QueryMetadata(
            query_type=__class__.__qualname__,
            execution_time_ms=elapsed_time,
            search_method="faiss" if self.use_faiss else "duckdb_vss"
        )

        return ChangeResultsSet(
            results=result_instances,
            query_text=self.text_query,
            query_year_from=self.year_from,
            query_year_to=self.year_to,
            sequential_only=self.sequential_only,
            metadata=metadata
        )
