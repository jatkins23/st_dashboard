"""Compute PCA whitening statistics for embedding normalization.

This script computes whitening transformation statistics that can be used
to improve retrieval quality by normalizing the embedding space.

Usage:
    # Compute whitening for a specific year
    python scripts/compute_whitening.py --universe lion --year 2020

    # Compute for all years
    python scripts/compute_whitening.py --universe lion --all-years

    # Use subset of data for faster computation
    python scripts/compute_whitening.py --universe lion --year 2020 --n-samples 10000

    # Custom number of PCA components
    python scripts/compute_whitening.py --universe lion --year 2020 --n-components 256

    # List existing statistics
    python scripts/compute_whitening.py --universe lion --list

    # Test retrieval improvement
    python scripts/compute_whitening.py --universe lion --year 2020 --test-retrieval --location 12345

Example workflow:
    1. Generate embeddings
    2. Compute whitening: python scripts/compute_whitening.py --universe lion --year 2020
    3. Query with reranking: Use EmbeddingLoader with use_whitening=True
"""

from argparse import ArgumentParser
from pathlib import Path
import logging

import numpy as np

from streettransformer import WhiteningTransform, Config, EmbeddingDB

logger = logging.getLogger(__name__)


def compute_statistics(
    universe_name: str,
    database_path: str,
    years: list[int] | None = None,
    stats_dir: str = './whitening_stats',
    cache_dir: str = './data/embedding_cache',
    n_components: int | None = None,
    n_samples: int | None = None,
    force_recompute: bool = False
):
    """Compute whitening statistics for specified years.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        years: List of years (None for all)
        stats_dir: Directory for statistics
        cache_dir: Directory for caches
        n_components: Number of PCA components
        n_samples: Number of samples to use
        force_recompute: Force recomputation
    """
    # Auto-detect years if not specified
    if years is None:
        config = Config(database_path=database_path, universe_name=universe_name)
        db = EmbeddingDB(config)
        years = db.get_years()
        logger.info(f"Auto-detected years: {years}")

    # Create whitening transform
    config = Config(database_path=database_path, universe_name=universe_name, stats_dir=stats_dir)
    whiten = WhiteningTransform(config)

    for year in years:
        logger.info(f"Computing whitening statistics for year {year}...")

        try:
            stats = whiten.compute_statistics(
                year=year,
                n_components=n_components,
                n_samples=n_samples,
                force_recompute=force_recompute
            )

            # Print variance explained
            var_ratio = stats.explained_variance / stats.explained_variance.sum()
            cumsum_var = np.cumsum(var_ratio)

            logger.info(f"✓ Computed statistics for year {year}")
            logger.info(f"  Samples: {stats.n_samples:,}")
            logger.info(f"  Components: {stats.n_components}")
            logger.info(f"  First 5 components explain: {cumsum_var[4]:.2%} variance")
            logger.info(f"  First 10 components explain: {cumsum_var[9]:.2%} variance")
            logger.info(f"  First 50 components explain: {cumsum_var[49]:.2%} variance")

        except Exception as e:
            logger.error(f"Failed to compute statistics for year {year}: {e}")


def list_statistics(universe_name: str, database_path: str, stats_dir: str = './whitening_stats'):
    """List all available whitening statistics.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        stats_dir: Directory for statistics
    """
    config = Config(database_path=database_path, universe_name=universe_name, stats_dir=stats_dir)
    whiten = WhiteningTransform(config)
    stats_list = whiten.list_statistics()

    if not stats_list:
        print(f"\nNo whitening statistics found for universe '{universe_name}'")
        return

    print(f"\n{'='*80}")
    print(f"Whitening Statistics for '{universe_name}'")
    print(f"{'='*80}\n")

    print(f"{'Year':<8} {'Samples':<12} {'Components':<12} {'Size (MB)':<12} {'Created':<20}")
    print("-" * 80)

    for info in stats_list:
        print(
            f"{info['year']:<8} {info['n_samples']:<12,} {info['n_components']:<12} "
            f"{info['file_size_mb']:<12.2f} {info['created_at'][:19]}"
        )

    print()


def test_retrieval(
    universe_name: str,
    database_path: str,
    year: int,
    location_id: int,
    stats_dir: str = './whitening_stats',
    k: int = 20
):
    """Test retrieval quality improvement with whitening.

    Args:
        universe_name: Universe name
        database_path: Path to DuckDB database
        year: Year
        location_id: Query location
        stats_dir: Directory for statistics
        k: Number of results
    """
    print(f"\n{'='*80}")
    print(f"Testing Retrieval Quality - Location {location_id}, Year {year}")
    print(f"{'='*80}\n")

    # Load query embedding
    config = Config(database_path=database_path, universe_name=universe_name)
    db = EmbeddingDB(config)

    # Get query embedding from database
    from streettransformer.database import get_connection
    with get_connection(database_path, read_only=True) as con:
        query_df = con.execute(f"""
            SELECT location_id, location_key, year, embedding
            FROM {universe_name}.image_embeddings
            WHERE location_id = {location_id}
                AND year = {year}
                AND embedding IS NOT NULL
        """).df()

    if query_df.empty:
        logger.error(f"No embedding found for location {location_id}, year {year}")
        return

    query_embedding = np.array(query_df.iloc[0]['embedding'])

    # Search without whitening
    logger.info("Searching without whitening...")
    results_original = db.search_similar(
        query_vector=query_embedding,
        limit=k,
        year=year
    )

    # Apply whitening and rerank
    logger.info("Applying whitening transformation...")
    config = Config(database_path=database_path, universe_name=universe_name, stats_dir=stats_dir)
    whiten = WhiteningTransform(config)

    try:
        # Load statistics
        whiten.load_statistics(year=year)

        # Rerank results
        results_whitened = whiten.rerank_results(
            query_vector=query_embedding,
            results=results_original,
            year=year,
            top_k=k
        )

        # Compare
        comparison = whiten.compare_retrieval(
            query_vector=query_embedding,
            results=results_original,
            year=year
        )

        # Print comparison
        print("Comparison Metrics:")
        print("-" * 80)
        print(f"Rank correlation: {comparison['rank_correlation']:.4f}")
        print(f"Mean similarity (original): {comparison['mean_similarity_original']:.4f}")
        print(f"Mean similarity (whitened): {comparison['mean_similarity_whitened']:.4f}")
        print(f"Std similarity (original): {comparison['std_similarity_original']:.4f}")
        print(f"Std similarity (whitened): {comparison['std_similarity_whitened']:.4f}")
        print(f"Std improvement: {comparison['std_improvement']:.2f}x")
        print()

        # Show top 10 results comparison
        print("Top 10 Results Comparison:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Location (Original)':<20} {'Sim (Orig)':<12} {'Location (Whitened)':<20} {'Sim (White)':<12}")
        print("-" * 80)

        for i in range(min(10, len(results_original))):
            orig_loc = results_original.iloc[i]['location_id']
            orig_sim = results_original.iloc[i]['similarity']
            white_loc = results_whitened.iloc[i]['location_id']
            white_sim = results_whitened.iloc[i]['similarity']

            # Mark if order changed
            marker = "←" if orig_loc != white_loc else ""

            print(
                f"{i+1:<6} {orig_loc:<20} {orig_sim:<12.4f} "
                f"{white_loc:<20} {white_sim:<12.4f} {marker}"
            )

        print()

    except FileNotFoundError:
        logger.error(f"Whitening statistics not found for year {year}")
        logger.info("Run compute_statistics() first")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser(description="Compute whitening statistics for embeddings")
    parser.add_argument('--db', '-d', type=str, required=True, help='Path to DuckDB database')
    parser.add_argument('--universe', '-u', type=str, required=True, help='Universe name')

    # Actions
    parser.add_argument('--list', action='store_true', help='List existing statistics')
    parser.add_argument('--test-retrieval', action='store_true', help='Test retrieval improvement')

    # Compute options
    parser.add_argument('--year', type=int, help='Specific year to compute')
    parser.add_argument('--years', nargs='+', type=int, help='Multiple years to compute')
    parser.add_argument('--all-years', action='store_true', help='Compute for all years')
    parser.add_argument('--n-components', type=int, help='Number of PCA components (default: vector_dim)')
    parser.add_argument('--n-samples', type=int, help='Number of samples to use (default: all)')

    # Directories
    parser.add_argument('--stats-dir', type=str, default='./whitening_stats',
                        help='Directory for statistics files')
    parser.add_argument('--cache-dir', type=str, default='./data/embedding_cache',
                        help='Directory for NPZ caches')

    # Flags
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of existing statistics')

    # Test options
    parser.add_argument('--location', type=int, help='Location ID for retrieval test')
    parser.add_argument('--k', type=int, default=20, help='Number of results for test')

    args = parser.parse_args()

    # List statistics
    if args.list:
        list_statistics(args.universe, args.db, args.stats_dir)
        return

    # Test retrieval
    if args.test_retrieval:
        if not args.year or not args.location:
            parser.error("--year and --location are required for retrieval test")

        test_retrieval(
            args.universe,
            args.db,
            args.year,
            args.location,
            stats_dir=args.stats_dir,
            k=args.k
        )
        return

    # Compute statistics
    if not args.year and not args.years and not args.all_years:
        parser.error("Either --year, --years, or --all-years is required")

    # Determine years to compute
    if args.all_years:
        years = None
    elif args.years:
        years = args.years
    else:
        years = [args.year]

    compute_statistics(
        universe_name=args.universe,
        database_path=args.db,
        years=years,
        stats_dir=args.stats_dir,
        cache_dir=args.cache_dir,
        n_components=args.n_components,
        n_samples=args.n_samples,
        force_recompute=args.force_recompute
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
