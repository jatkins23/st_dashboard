import logging
import hashlib

from ..base_query import BaseQuery
from ..mixins import TextQueryMixin
from ..results_sets import StateResultsSet

from ... import FAISSIndexer
from ... import WhiteningTransform


class StateDescriptionQuery(TextQueryMixin, BaseQuery):
    """
    Compares a text string to a state Image
    """
    # TODO: Write descrpition
    # clip_encoder: ClipEncoder # TODO: Move to TextTranslationMixin? I think I did this 
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
