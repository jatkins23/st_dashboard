#!/usr/bin/env python3
"""Basic test of streettransformer package with existing database."""

import sys
sys.path.insert(0, 'src')

from streettransformer import Config, EmbeddingDB
import numpy as np

def main():
    print("Testing StreetTransformer with existing database...")

    # Configure to use existing st_preprocessing database
    config = Config(
        database_path="/Users/jon/code/st_preprocessing/core.ddb",
        universe_name="lion"
    )

    print(f"✓ Config created: {config.database_path}")
    print(f"  Universe: {config.universe_name}")
    print(f"  Vector dim: {config.vector_dim}")

    # Test database connection
    db = EmbeddingDB(config)
    print("✓ EmbeddingDB initialized")

    # Get stats
    try:
        count = db.get_embedding_count()
        years = db.get_years()
        print(f"✓ Database connected successfully")
        print(f"  Total embeddings: {count:,}")
        print(f"  Years available: {years}")
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return 1

    # Test search if we have data
    if count > 0 and years:
        print(f"\nTesting search for year {years[0]}...")
        query_vector = np.random.rand(config.vector_dim).astype(np.float32)

        try:
            results = db.search_similar(query_vector, limit=5, year=years[0])
            print(f"✓ Search successful")
            print(f"  Found {len(results)} results")
            if not results.empty:
                print(f"  Top result: {results.iloc[0]['location_key']} "
                      f"(similarity: {results.iloc[0]['similarity']:.4f})")
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return 1

    print("\n✅ All basic tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
