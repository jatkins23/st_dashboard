import logging
import time

import numpy as np
import pandas as pd

from ..base_query import BaseQuery
from ..mixins import StateMixin
from ..results_sets import ChangeResultsSet
from ..results_instances import ChangeResultInstance
from ..metadata import QueryMetadata


class StateDissimilarityQuery(BaseQuery, StateMixin):
    """Longitudinal dissimilarity query - find when a location changed most.

    Analyzes a single location across a time range to identify when it changed
    most dramatically. Useful for detecting transformation events.

    Attributes:
        location_id: The location to analyze
        year_start: Beginning of time range
        year_end: End of time range
        comparison_strategy: How to compare years
            - 'sequential': Compare year N to N+1 (default, fastest)
            - 'all_pairs': Compare all year combinations (comprehensive)
            - 'baseline': Compare all years to first year (detect cumulative change)
    """
    year_start: int
    year_end: int
    comparison_strategy: str = 'sequential'

    def search(self) -> ChangeResultsSet:
        """Execute dissimilarity search."""
        start_time = time.perf_counter()
        logger = logging.getLogger(__name__)

        # 1. Fetch all embeddings for this location within year range
        embeddings_df = self._fetch_location_embeddings()

        if embeddings_df.empty:
            logger.error(f"No embeddings found for location {self.location_id} in range {self.year_start}-{self.year_end}")
            return ChangeResultsSet(results=[], query_location_id=self.location_id)

        # 2. Compute pairwise dissimilarities based on strategy
        if self.comparison_strategy == 'sequential':
            dissimilarities = self._compute_sequential_dissimilarity(embeddings_df)
        elif self.comparison_strategy == 'all_pairs':
            dissimilarities = self._compute_all_pairs_dissimilarity(embeddings_df)
        else:  # baseline
            dissimilarities = self._compute_baseline_dissimilarity(embeddings_df)

        if dissimilarities.empty:
            logger.warning(f"No dissimilarities computed for location {self.location_id}")
            return ChangeResultsSet(results=[], query_location_id=self.location_id)

        # 3. Sort by dissimilarity (highest = most changed)
        dissimilarities = dissimilarities.sort_values('dissimilarity', ascending=False)

        # 4. Limit results
        dissimilarities = dissimilarities.head(self.limit)

        # 5. Convert to ChangeResultsSet format
        results_dict = dissimilarities.to_dict(orient='records')
        result_instances = [ChangeResultInstance.model_validate(row) for row in results_dict]

        elapsed_time = (time.perf_counter() - start_time) * 1000

        metadata = QueryMetadata(
            query_type=__class__.__qualname__,
            execution_time_ms=elapsed_time,
            search_method=f"dissimilarity_{self.comparison_strategy}"
        )

        return ChangeResultsSet(
            results=result_instances,
            query_location_id=self.location_id,
            query_year_from=self.year_start,
            query_year_to=self.year_end,
            metadata=metadata
        )

    def _fetch_location_embeddings(self) -> pd.DataFrame:
        """Fetch all embeddings for the location within year range."""
        query_sql = f"""
            SELECT location_id, location_key, year, media_type, path, embedding
            FROM {self.get_universe_table('media_embeddings')}
            WHERE location_id = '{self.location_id}'
                AND year BETWEEN {self.year_start} AND {self.year_end}
                AND embedding IS NOT NULL
            ORDER BY year
        """
        return self.execute_query(query_sql)

    def _compute_sequential_dissimilarity(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Compare year N to N+1."""
        results = []
        years = sorted(embeddings_df['year'].unique())

        for i in range(len(years) - 1):
            year_from = years[i]
            year_to = years[i + 1]

            emb_from = embeddings_df[embeddings_df['year'] == year_from].iloc[0]['embedding']
            emb_to = embeddings_df[embeddings_df['year'] == year_to].iloc[0]['embedding']

            # Compute cosine dissimilarity (1 - cosine_similarity)
            vec_from = np.array(emb_from)
            vec_to = np.array(emb_to)
            cosine_sim = np.dot(vec_from, vec_to) / (np.linalg.norm(vec_from) * np.linalg.norm(vec_to))
            dissimilarity = 1 - cosine_sim

            results.append({
                'location_id': self.location_id,
                'location_key': embeddings_df.iloc[0]['location_key'],
                'year_from': year_from,
                'year_to': year_to,
                'dissimilarity': dissimilarity,
                'distance': dissimilarity  # Compatibility with ChangeResultInstance
            })

        return pd.DataFrame(results)

    def _compute_all_pairs_dissimilarity(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Compare all year combinations."""
        results = []
        years = sorted(embeddings_df['year'].unique())

        for i, year_from in enumerate(years):
            for year_to in years[i+1:]:
                emb_from = embeddings_df[embeddings_df['year'] == year_from].iloc[0]['embedding']
                emb_to = embeddings_df[embeddings_df['year'] == year_to].iloc[0]['embedding']

                vec_from = np.array(emb_from)
                vec_to = np.array(emb_to)
                cosine_sim = np.dot(vec_from, vec_to) / (np.linalg.norm(vec_from) * np.linalg.norm(vec_to))
                dissimilarity = 1 - cosine_sim

                results.append({
                    'location_id': self.location_id,
                    'location_key': embeddings_df.iloc[0]['location_key'],
                    'year_from': year_from,
                    'year_to': year_to,
                    'dissimilarity': dissimilarity,
                    'distance': dissimilarity
                })

        return pd.DataFrame(results)

    def _compute_baseline_dissimilarity(self, embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Compare all years to first year."""
        results = []
        years = sorted(embeddings_df['year'].unique())

        if len(years) < 2:
            return pd.DataFrame()

        baseline_year = years[0]
        baseline_emb = embeddings_df[embeddings_df['year'] == baseline_year].iloc[0]['embedding']
        baseline_vec = np.array(baseline_emb)

        for year_to in years[1:]:
            emb_to = embeddings_df[embeddings_df['year'] == year_to].iloc[0]['embedding']
            vec_to = np.array(emb_to)

            cosine_sim = np.dot(baseline_vec, vec_to) / (np.linalg.norm(baseline_vec) * np.linalg.norm(vec_to))
            dissimilarity = 1 - cosine_sim

            results.append({
                'location_id': self.location_id,
                'location_key': embeddings_df.iloc[0]['location_key'],
                'year_from': baseline_year,
                'year_to': year_to,
                'dissimilarity': dissimilarity,
                'distance': dissimilarity
            })

        return pd.DataFrame(results)
