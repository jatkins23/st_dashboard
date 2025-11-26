# StreetTransformer

Vector embeddings for street imagery analysis with DuckDB and FAISS.

## Features

- **DuckDB Storage**: Efficient vector storage with VSS extension
- **FAISS Indexing**: Fast approximate nearest neighbor search
- **PCA Whitening**: Improved retrieval quality through whitening transformation
- **NPZ Caching**: Fast embedding loading with compressed caches
- **CLI Tools**: Command-line interface for querying and generating embeddings

## Installation

```bash
# Basic installation
pip install -e .

# With FAISS support
pip install -e ".[faiss]"

# With CLI tools
pip install -e ".[cli]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Using with existing st_preprocessing database

```python
from streettransformer import Config, EmbeddingDB

# Point to your st_preprocessing database
config = Config(
    database_path="/Users/jon/code/st_preprocessing/data.db",
    universe_name="lion"
)

# Query embeddings
db = EmbeddingDB(config)
import numpy as np

query_vector = np.random.rand(512)  # Your query embedding
results = db.search_similar(query_vector, limit=10, year=2020)
print(results)
```

### Fast search with FAISS

```python
from streettransformer import Config, FAISSIndexer

config = Config(database_path="data.db", universe_name="lion")
indexer = FAISSIndexer(config)

# Build index (first time only)
indexer.build_index(year=2020, index_type='hnsw')

# Search
results = indexer.search(query_vector, k=10, year=2020)
```

### Whitening for better retrieval

```python
from streettransformer import Config, WhiteningTransform

config = Config(database_path="data.db", universe_name="lion")
whiten = WhiteningTransform(config)

# Compute statistics (first time only)
whiten.compute_statistics(year=2020)

# Rerank results
reranked = whiten.rerank_results(
    query_vector=query_vector,
    results=results,
    year=2020,
    top_k=10
)
```

## Architecture

```
streettransformer/
├── config.py          # Configuration
├── database.py        # DuckDB connection
├── embedding_db.py    # Core vector storage
├── npz_cache.py       # NPZ caching
├── faiss_index.py     # FAISS indexing
└── whitening.py       # PCA whitening
```

## Database Schema

The package expects these tables in your DuckDB database:

```sql
{universe}.image_embeddings (
    location_id BIGINT,
    location_key VARCHAR,
    year INTEGER,
    image_path VARCHAR,
    embedding FLOAT[512],
    PRIMARY KEY (location_key, year)
)

{universe}.change_vectors (
    location_id BIGINT,
    location_key VARCHAR,
    year_from INTEGER,
    year_to INTEGER,
    delta FLOAT[512],
    PRIMARY KEY (location_key, year_from, year_to)
)
```

## License

MIT
