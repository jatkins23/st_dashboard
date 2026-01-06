# StreetTransformer Module ReadMe
This Directory contains the core search logic for the StreetTransformer. It represents the bridge between the pre-processing logic in `st_preprocessing` that creates the core database and the 

It has two main functions right now:
1. Create the embeddings and indices that allow for accurate and efficient querying of the data contained in the preprocessing database.
2. Define the retrieval logic that allows users to query these data from the dashboard (see dashboard Readme)

## Query
Currently the query logic is held in `query`, where each type of query has its own class.

The design principle is that each stage of the querying process is represented by its own object that shares as much code as possible with objects from different query pipelines at the same stage.

For example, the results of `ChangeDescriptionQuery` and `ImageSimilarityQuery` are both `ResultInstance`s.

Note (repeated below): The queries here are defined to interact with a sql-based vector database defined in `db`, with a fairly standard but under-optimized structure. The transition to the flat `npy` files (see image_retrieval) will require reorganizing each query to interact with these flat files, and thus it is an ongoing question as to how much of the structure of the objects contained here need to be modified as well.

### Mixins
Queries can differ in data_type (e.g. Image, Text, etc.), comparison_type (State vs. Change) or query_type (e.g. Similarity, Dissimilarity, etc.) and combinations of each of these can represent differently defined query logic, even while much of the underlying logic is the same. As such, much of this shared logic is stored in Mixins allowing shared code across these three different dimensions.


## DB
- This is my implementation that is being superceded by the processes in `image_retrieval`

## Next Steps:
- Goals are to translate the query logic defined in `image_retrieval`, which relies on pre-computed indices and rankings stored in flat-files (as defined in `pipeline`) into the paradigm as it is defined in `queries`. 
- The queries as currently written are defined to interact with a sql-based vector database defined in `db`, with a fairly standard but under-optimized structure. The transition to the flat `npy` files (see image_retrieval) will require reorganizing each query to interact with these flat files, and thus it is an ongoing question as to how much of the structure of the objects contained here need to be modified as well.