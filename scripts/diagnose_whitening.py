#!/usr/bin/env python
"""Diagnose whitening statistics issues."""

import sys
from pathlib import Path
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def diagnose_stats(stats_dir: str, universe: str, year: int):
    """Diagnose whitening statistics for issues.

    Args:
        stats_dir: Directory containing statistics
        universe: Universe name
        year: Year to check
    """
    stats_path = Path(stats_dir)
    stats_files = list(stats_path.glob(f"{universe}_{year}_*.npz"))

    if not stats_files:
        print(f"❌ No stats files found for {universe} year {year} in {stats_dir}")
        return

    stats_file = stats_files[0]
    print(f"📊 Analyzing: {stats_file.name}\n")

    data = np.load(stats_file)
    explained_var = data['explained_variance']
    components = data['components']
    mean = data['mean']
    n_components = data['n_components']
    n_samples = data['n_samples']

    print(f"{'='*60}")
    print(f"Statistics Summary")
    print(f"{'='*60}")
    print(f"  Samples:     {n_samples:,}")
    print(f"  Components:  {n_components}")
    print(f"  Vector dim:  {len(mean)}")
    print()

    print(f"{'='*60}")
    print(f"Explained Variance Analysis")
    print(f"{'='*60}")
    print(f"  Min:  {explained_var.min():.10f}")
    print(f"  Max:  {explained_var.max():.6f}")
    print(f"  Mean: {explained_var.mean():.6f}")
    print()

    # Check for issues
    zero_count = (explained_var == 0).sum()
    near_zero_count = (explained_var < 1e-10).sum()

    if zero_count > 0:
        print(f"⚠️  ISSUE: {zero_count} components with ZERO variance")
        zero_indices = np.where(explained_var == 0)[0]
        print(f"   Zero variance at indices: {zero_indices[:20].tolist()}")
        print()
        print("   Possible causes:")
        print("   - Requested more components than data dimensionality")
        print(f"   - Max useful components ≈ min(n_samples, vector_dim)")
        print(f"   - Your case: min({n_samples}, {len(mean)}) = {min(n_samples, len(mean))}")
        print(f"   - You requested: {n_components}")
        print()

    if near_zero_count > zero_count:
        print(f"⚠️  WARNING: {near_zero_count - zero_count} components with near-zero variance (<1e-10)")
        print()

    # Variance distribution
    cumsum_var = np.cumsum(explained_var) / explained_var.sum()
    print(f"Cumulative variance explained:")
    for n in [1, 5, 10, 50, 100, min(256, len(cumsum_var)-1)]:
        if n < len(cumsum_var):
            print(f"  First {n:3d} components: {cumsum_var[n-1]:.2%}")
    print()

    # Check mean vector
    mean_norm = np.linalg.norm(mean)
    mean_nonzero = np.count_nonzero(mean)
    print(f"Mean vector:")
    print(f"  Norm:     {mean_norm:.6f}")
    print(f"  Non-zero: {mean_nonzero}/{len(mean)}")
    print()

    # Recommendations
    if zero_count > 0:
        recommended_components = min(n_samples, len(mean)) - 10
        print(f"{'='*60}")
        print(f"💡 Recommendation")
        print(f"{'='*60}")
        print(f"Reduce n_components to: {recommended_components}")
        print(f"Recompute statistics with:")
        print(f"  python scripts/compute_whitening.py \\")
        print(f"    --db <path> --universe {universe} --year {year} \\")
        print(f"    --n-components {recommended_components} --force-recompute")
        print()


def main():
    parser = ArgumentParser(description="Diagnose whitening statistics issues")
    parser.add_argument('--db', '-d', type=str,
                        help='Path to DuckDB database (or set ST_DATABASE_PATH env var, not used but kept for consistency)')
    parser.add_argument('--stats-dir', type=str, default='./data/whitening_stats',
                        help='Statistics directory')
    parser.add_argument('--universe', '-u', type=str, required=True,
                        help='Universe name')
    parser.add_argument('--year', '-y', type=int, required=True,
                        help='Year to check')

    args = parser.parse_args()
    diagnose_stats(args.stats_dir, args.universe, args.year)


if __name__ == '__main__':
    main()
