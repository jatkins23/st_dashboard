This is supposed to be the union of Query and ImageRetrieval


Plan:
- utilize the ResultInstances (or an encapsulating class) to replace much of the functionality currently contained in VectorDB
    - `SearchHit` -> `StateResultInstance` (or something like `StateSimilarityInstanceVecDB` ??)
- rewrite each of the query types in `image_retrieval.query` into the separate modules in `query.queries.[query_name]`