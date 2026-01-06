import logging
import time

import numpy as np
import pandas as pd

from ..base_query import BaseQuery
from ..mixins import StateMixin
from ..results_sets import StateResultsSet
from ..results_instances import StateResultInstance
from ..metadata import QueryMetadata

from ... import FAISSIndexer
from ... import WhiteningTransform


class StateSimilarityQuery(BaseQuery, StateMixin):
    """
    Compares Images -> Images based on similarity
    """
    def search(self) -> StateResultsSet:
        start_time = time.perf_counter()

        results_dict = self._execute_search().to_dict(orient='records')
        results = [StateResultInstance.model_validate(row) for row in results_dict]

        elapsed_time = (time.perf_counter() - start_time) * 1000

        metadata = QueryMetadata(
            query_type = __class__.__qualname__,
            execution_time_ms = elapsed_time,
            search_method = "faiss" if self.use_faiss else "duckdb_vss"
        )

        return StateResultsSet(
            results=results,
            query_location_id=self.location_id,
            query_year=self.year,
            target_years=self.target_years,
            metadata=metadata
        )

    def _execute_search(self):
        """Execute search from database"""
        logger = logging.getLogger(__name__)

        # Build media_type filter - default to 'image' if not specified
        media_types = getattr(self, 'media_types', None) or ['image']
        if isinstance(media_types, str):
            media_types = [media_types]

        media_type_filter = "', '".join(media_types)

        query_df = self.execute_query(f"""
            SELECT location_id, location_key, year, media_type, path, embedding
            FROM {self.get_universe_table('media_embeddings')}
            WHERE location_id = '{self.location_id}'
                AND year = {self.year}
                AND media_type IN ('{media_type_filter}')
                AND embedding IS NOT NULL
            ;
        """)

        if query_df.empty:
            logger.error(f"No embedding found for location {self.location_id} year {self.year} with media types {media_types}")
            return pd.DataFrame()

        if query_df.shape[0] > 1:
            logger.warning(f"More than one embedding found for location {self.location_id} year {self.year}\n\tDefaulting to first one.")

        query_vec = np.array(query_df.iloc[0]['embedding'])

        if self.use_faiss:
            indexer = FAISSIndexer(self.config)
            results = indexer.search(
                query_vector=query_vec,
                k = self.limit * 2, # See below for why 2
                year = self.target_years[0] if self.target_years else self.year # TODO: fix years to be a list . Also this should search all years!
            )

            # Filter results by media_type if FAISS index contains multiple types
            if 'media_type' in results.columns:
                results = results[results['media_type'].isin(media_types)]

        # Remove self from results
        if self.remove_self:
            results = results[results['location_id'] != self.location_id]
        results = results.head(self.limit)

        # Apply whitening reranking if requested
        if self.use_whitening and not results.empty:
            whiten = WhiteningTransform(self.config)
            # Use first media type for whitening (assuming whitening stats are per media_type)
            results = whiten.rerank_results(
                query_vector=query_vec,
                results=results,
                year=self.target_years[0] if self.target_years else self.year, # TODO: again make sure target_year takes a list
                media_type=media_types[0] if len(media_types) == 1 else None,
                top_k=self.limit
            )

        return results
