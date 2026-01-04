import time
from typing import Any, Optional
import logging
from abc import ABC
from pydantic import BaseModel, ConfigDict
import hashlib

import numpy as np
import pandas as pd

from ..mixins import SearchMethodMixin, DatabaseMixin, ChangeMixin, StateMixin, TextQueryMixin
from .results_sets import QueryResultsSet, StateResultsSet, ChangeResultsSet
from .results_instances import StateResultInstance, ChangeResultInstance
from .metadata import QueryMetadata

from ... import FAISSIndexer
from ... import WhiteningTransform

class BaseQuery(DatabaseMixin, SearchMethodMixin, BaseModel):
    # use_faiss: bool = True
    # use_whitening: bool = False
    distance_metric: str = 'cosine'

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def search(self, **kwargs: Any) -> QueryResultsSet:
        ...

    def get_cache_key(self) -> str:

        """Get unique cache key for this query.

        Returns:
            String like: loc_123_change2015to2020_n10
        """
        ...

class ImageToImageStateQuery(BaseQuery, StateMixin):
    def search(self)  -> StateResultsSet:
        start_time = time.perf_counter()
        results_dict = self._execute_search().to_dict(orient='records')
        results = [StateResultInstance.model_validate(row) for row in results_dict]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        metadata = QueryMetadata(
            query_type = __class__.__qualname__,
            execution_time_ms = elapsed_time * 1000,
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
            

class ImageToImageChangeQuery(BaseQuery, ChangeMixin):
    def search(self):
        start_time = time.perf_counter()
        results_dict = self._execute_search().to_dict(orient='records')
        results = [ChangeResultInstance.model_validate(row) for row in results_dict]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        metadata = QueryMetadata(
            query_type = __name__,
            execution_time_ms = elapsed_time * 1000,
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
    
    def _execute_search(self) -> dict:
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


class TextToImageStateQuery(TextQueryMixin, BaseQuery):
    # clip_encoder: ClipEncoder # TODO: Move to TextTranslationMixin?
    # text_query: str
    # target_years: Optional[list[str]] = None

    # Allow Arbitrary Types? -- already implemented
    def execute(self) -> StateResultsSet:
        logger = logging.getLogger(__name__)

        # Encode text query
        if self.clip_encoder is None:
            raise ValueError(
                "clip_encoder is required for text queries. "
                "Create a CLIPEncoder instance and pass it to the query."
            )
        logger.info(f"Encoding text query: '{self.text_query}'")
        query_embedding = self.clip_encoder.encode(self.text_query)
        print(query_embedding)
        
        # Search using FAISS if requested
        if self.use_faiss:
            indexer = FAISSIndexer(self.config)
            results = indexer.search(
                query_vector=query_embedding,
                k=self.limit,
                year= 2018 # TODO: remvoe year from FAISSIndexer
            )
        else:
            # Use database search
            results = self.db.search_similar(
                query_vector=query_embedding,
                limit=self.limit,
                year=self.year
            )

        # Apply Whitening reranking
        if self.use_whitening and not results.empty:
            whiten = WhiteningTransform(self.config)
            results = whiten.rerank_results(
                query_vector=query_embedding,
                results=results,
                year=self.target_years[0] if self.target_years else None,
                top_k = self.limit
            )

        return results

    def get_cache_key(self) -> str:
        """Get unique cache key for this query.

        Uses MD5 hash of text query to keep key manageable.

        Returns:
            String like: text_a3f2b8c1_y2020_fw_n10
        """
        text_hash = hashlib.md5(self.text_query.encode()).hexdigest()[:8]
        year_str = f"y[{'_'.join(self.target_years)}]" if self.year else "all_years"
        method = "f" if self.use_faiss else "d"
        method += "w" if self.use_whitening else ""
        return f"text_{text_hash}_{year_str}_{method}_n{self.limit}"


class TextToImageChangeQuery(TextQueryMixin, ChangeMixin, BaseQuery):
    """Text-to-image change detection query.

    Combines text-based querying with temporal change detection.
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
        start_time = time.perf_counter()
        results_dict = results.to_dict(orient='records')
        result_instances = [ChangeResultInstance.model_validate(row) for row in results_dict]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        metadata = QueryMetadata(
            query_type=__class__.__qualname__,
            execution_time_ms=elapsed_time * 1000,
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


class ImageToImageDissimilarityQuery(BaseQuery, StateMixin):
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
        start_time = time.perf_counter()
        results_dict = dissimilarities.to_dict(orient='records')
        result_instances = [ChangeResultInstance.model_validate(row) for row in results_dict]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        metadata = QueryMetadata(
            query_type=__class__.__qualname__,
            execution_time_ms=elapsed_time * 1000,
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
