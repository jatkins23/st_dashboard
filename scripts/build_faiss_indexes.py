"""Build FAISS indexes for fast similarity search.

This script builds FAISS indexes from embeddings stored in DuckDB,
enabling much faster similarity search compared to database queries.

Usage:
    # Build HNSW index for a specific year
    python scripts/build_faiss_indexes.py --universe lion --year 2020 --index-type hnsw

    # Build for all years
    python scripts/build_faiss_indexes.py --universe lion --all-years --index-type hnsw

    # Build multiple index types
    python scripts/build_faiss_indexes.py --universe lion --year 2020 --index-types hnsw ivf_flat

    # List existing indexes
    python scripts/build_faiss_indexes.py --universe lion --list

    # Benchmark index performance
    python scripts/build_faiss_indexes.py --universe lion --year 2020 --benchmark

Example workflow:
    1. Generate embeddings: python scripts/generate_embeddings.py --universe lion --year 2020
    2. Build cache: python scripts/build_faiss_indexes.py --universe lion --year 2020 --build-cache
    3. Build index: python scripts/build_faiss_indexes.py --universe lion --year 2020 --index-type hnsw
    4. Query: Use updated EmbeddingLoader with use_faiss=True
"""

from argparse import ArgumentParser
from pathlib import Path
import logging

from streettransformer import FAISSIndexer, NPZCache, Config, EmbeddingDB

logger = logging.getLogger(__name__)


def build_indexes(
    universe_name: str,
    database_path: str,
    years: list[int] | None = None,
    index_types: list[str] = ['hnsw'],
    index_dir: str = './data/faiss_indexes',
    cache_dir: str = './data/embedding_cache',
    force_rebuild: bool = False,
    build_cache_first: bool = True
):
    """Build FAISS indexes for specified years.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        years: List of years (None for all)
        index_types: List of index types to build
        index_dir: Directory for indexes
        cache_dir: Directory for caches
        force_rebuild: Force rebuild existing indexes
        build_cache_first: Build NPZ cache before index
    """
    # Auto-detect years if not specified
    if years is None:
        config = Config(database_path=database_path, universe_name=universe_name)
        db = EmbeddingDB(config)
        years = db.get_years()
        logger.info(f"Auto-detected years: {years}")

    # Create config
    config = Config(
        database_path=database_path,
        universe_name=universe_name,
        cache_dir=cache_dir,
        index_dir=index_dir
    )

    # Build cache first if requested
    if build_cache_first:
        logger.info("Building NPZ caches...")
        cache = NPZCache(config)
        for year in years:
            cache.build_from_db(year=year, force_rebuild=force_rebuild)

    # Build indexes
    indexer = FAISSIndexer(config)

    for year in years:
        for index_type in index_types:
            logger.info(f"Building {index_type} index for year {year}...")

            try:
                index_path = indexer.build_index(
                    year=year,
                    index_type=index_type,
                    force_rebuild=force_rebuild,
                    use_cache=True
                )

                info = indexer.get_index_info(year, index_type)
                logger.info(
                    f"✓ Built {index_type} index: {info.num_vectors} vectors, "
                    f"{info.file_size_mb:.2f} MB"
                )

            except Exception as e:
                logger.error(f"Failed to build {index_type} index for year {year}: {e}")


def list_indexes(universe_name: str, database_path: str, index_dir: str = './data/faiss_indexes'):
    """List all available indexes.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        index_dir: Directory for indexes
    """
    config = Config(database_path=database_path, universe_name=universe_name, index_dir=index_dir)
    indexer = FAISSIndexer(config)
    indexes = indexer.list_indexes()

    if not indexes:
        print(f"\nNo indexes found for universe '{universe_name}'")
        return

    print(f"\n{'='*80}")
    print(f"FAISS Indexes for '{universe_name}'")
    print(f"{'='*80}\n")

    print(f"{'Year':<8} {'Type':<12} {'Vectors':<12} {'Size (MB)':<12} {'Created':<20}")
    print("-" * 80)

    for info in indexes:
        print(
            f"{info.year:<8} {info.index_type:<12} {info.num_vectors:<12,} "
            f"{info.file_size_mb:<12.2f} {info.created_at[:19]}"
        )

    print()


def benchmark_indexes(
    universe_name: str,
    database_path: str,
    year: int,
    index_types: list[str] = ['hnsw', 'ivf_flat'],
    index_dir: str = './data/faiss_indexes',
    n_queries: int = 100,
    k: int = 10
):
    """Benchmark index performance.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        year: Year to benchmark
        index_types: Index types to benchmark
        index_dir: Directory for indexes
        n_queries: Number of queries
        k: Number of results per query
    """
    config = Config(database_path=database_path, universe_name=universe_name, index_dir=index_dir)
    indexer = FAISSIndexer(config)

    print(f"\n{'='*80}")
    print(f"FAISS Index Benchmark - Year {year}")
    print(f"{'='*80}\n")
    print(f"Queries: {n_queries}, k: {k}\n")

    results = []

    for index_type in index_types:
        try:
            info = indexer.get_index_info(year, index_type)
            if not info:
                logger.warning(f"Index not found: {index_type} for year {year}")
                continue

            logger.info(f"Benchmarking {index_type}...")
            stats = indexer.benchmark_search(
                year=year,
                index_type=index_type,
                n_queries=n_queries,
                k=k
            )

            results.append({
                'index_type': index_type,
                **stats
            })

        except Exception as e:
            logger.error(f"Failed to benchmark {index_type}: {e}")

    # Print results
    if results:
        print(f"{'Index Type':<12} {'Mean (ms)':<12} {'Median (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12}")
        print("-" * 80)

        for r in results:
            print(
                f"{r['index_type']:<12} {r['mean_ms']:<12.2f} {r['median_ms']:<12.2f} "
                f"{r['p95_ms']:<12.2f} {r['p99_ms']:<12.2f}"
            )
        print()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser(description="Build FAISS indexes for embeddings")
    parser.add_argument('--db', '-d', type=str, required=True, help='Path to DuckDB database')
    parser.add_argument('--universe', '-u', type=str, required=True, help='Universe name')

    # Actions
    parser.add_argument('--list', action='store_true', help='List existing indexes')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark index performance')

    # Build options
    parser.add_argument('--year', type=int, help='Specific year to build')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years to build')
    parser.add_argument('--all-years', action='store_true', help='Build for all years')
    parser.add_argument('--index-type', type=str, default='hnsw',
                        choices=['flat', 'ivf_flat', 'ivf_pq', 'hnsw'],
                        help='Index type to build')
    parser.add_argument('--index-types', nargs='+',
                        choices=['flat', 'ivf_flat', 'ivf_pq', 'hnsw'],
                        help='Multiple index types to build')

    # Directories
    parser.add_argument('--index-dir', type=str, default='./data/faiss_indexes',
                        help='Directory for index files')
    parser.add_argument('--cache-dir', type=str, default='./embedding_cache',
                        help='Directory for NPZ caches')

    # Flags
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild existing indexes')
    parser.add_argument('--build-cache', action='store_true',
                        help='Build NPZ cache before indexing')
    parser.add_argument('--skip-cache', action='store_true',
                        help='Skip using NPZ cache (load from DB)')

    # Benchmark options
    parser.add_argument('--n-queries', type=int, default=100,
                        help='Number of queries for benchmark')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of results per query')

    args = parser.parse_args()

    # List indexes
    if args.list:
        list_indexes(args.universe, args.db, args.index_dir)
        return

    # Benchmark indexes
    if args.benchmark:
        if not args.year:
            parser.error("--year is required for benchmark")

        index_types = args.index_types if args.index_types else [args.index_type]
        benchmark_indexes(
            args.universe,
            args.db,
            args.year,
            index_types=index_types,
            index_dir=args.index_dir,
            n_queries=args.n_queries,
            k=args.k
        )
        return

    # Build indexes
    if not args.year and not args.years and not args.all_years:
        parser.error("Either --year, --years, or --all-years is required")

    # Determine years to build
    if args.all_years:
        years = None
    elif args.years:
        years = args.years
    else:
        years = [args.year]

    index_types = args.index_types if args.index_types else [args.index_type]

    build_indexes(
        universe_name=args.universe,
        database_path=args.db,
        years=years,
        index_types=index_types,
        index_dir=args.index_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
        build_cache_first=args.build_cache or not args.skip_cache
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
