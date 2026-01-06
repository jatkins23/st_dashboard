import logging
import time

import numpy as np
import pandas as pd

from ..base_query import BaseQuery
from ..mixins import ChangeMixin
from ..results_sets import ChangeResultsSet
from ..results_instances import ChangeResultInstance
from ..metadata import QueryMetadata


class ChangeSimilarityQuery(BaseQuery, ChangeMixin):
    """
    Compares Longitudinal Image Pairs to other sets of longitudinal image pairs
    
    :var location_id: Description
    :vartype location_id: Literal['{self.location_id}']
    """
    def search(self) -> ChangeResultsSet:
        start_time = time.perf_counter()

        results_dict = self._execute_search().to_dict(orient='records')
        results = [ChangeResultInstance.model_validate(row) for row in results_dict]

        elapsed_time = (time.perf_counter() - start_time) * 1000

        metadata = QueryMetadata(
            query_type = __name__,
            execution_time_ms = elapsed_time,
            search_method = self.get_search_method_name()
        )

        return ChangeResultsSet(
            results=results,
            query_location_id=self.location_id,
            query_year_from=self.year_from,
            query_year_to=self.year_to,
            target_years=self.target_years,
            sequential=self.sequential_only,
            metadata=metadata
        )

    def _execute_search(self) -> pd.DataFrame:
        logger = logging.getLogger(__name__)
        query_df = self.execute_query(f"""
            SELECT location_id, location_key, year, embedding
            FROM {self.get_universe_table('media_embeddings')}
            WHERE location_id = '{self.location_id}'
                AND year IN ({self.year_from}, {self.year_to})
                AND embedding IS NOT NULL
        ;""")

        emb_from = query_df[query_df['year'] == self.year_from]
        emb_to = query_df[query_df['year'] == self.year_to]

        if emb_from.empty or emb_to.empty:
            logger.error(f"Missing embeddings for location {self.location_id}")
            return pd.DataFrame()

        if emb_from.shape[0] > 1 or emb_to.shape[0] > 1:
            logger.warning(f"More than one embedding found for location {self.location_id}.\n\tDefaulting to first. Confirm this is correct.")

        # Compute query delta
        vec_from = np.array(emb_from.iloc[0]['embedding'])
        vec_to = np.array(emb_to.iloc[0]['embedding'])
        query_delta = vec_to - vec_from

        # Normalize
        norm = np.linalg.norm(query_delta)
        if norm > 0:
            query_delta = query_delta / norm

        # Search for similar changes
        results = self.db.search_change_vectors(
            query_delta=query_delta,
            #limit=self.limit + 1, # TODO: This could be a problem. It adds one to account for self but if there is more than one instance of self (in multiple years), that could be a sneaky error
                                # Can actually just grab more if we want and get rid of them later
            limit = self.limit * 2,  # Multiplying by two for safety
            year_from = self.year_from,
            year_to = self.year_to
            # TODO: target years and sequential??
        )

        if self.remove_self:
            results = results[results['location_id'] != self.location_id]
            results = results.head(self.limit) #

        return results
