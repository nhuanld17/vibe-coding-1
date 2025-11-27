"""
Benchmark script for measuring with_vectors performance impact.

This script compares search performance with and without vector retrieval
to quantify the overhead of retrieving embeddings from Qdrant.

Usage:
    python scripts/benchmark_with_vectors.py

Requirements:
    - Qdrant running on localhost:6333
    - Test data in missing_persons and found_persons collections

Author: AI Face Recognition Team
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass  # Fallback to default encoding

import time
import numpy as np
from typing import List, Dict
from statistics import mean, median, stdev
from loguru import logger

from services.vector_db import VectorDatabaseService


def create_test_embedding() -> np.ndarray:
    """Create a random normalized test embedding."""
    embedding = np.random.rand(512).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def benchmark_search(
    vector_db: VectorDatabaseService,
    collection_name: str,
    num_iterations: int = 50,
    search_limit: int = 20,
    with_vectors: bool = False
) -> List[float]:
    """
    Benchmark search operation.
    
    Args:
        vector_db: VectorDatabaseService instance
        collection_name: Collection to search
        num_iterations: Number of search iterations
        search_limit: Number of results per search
        with_vectors: Whether to retrieve vectors
        
    Returns:
        List of latencies (in milliseconds) for each iteration
    """
    latencies = []
    
    for i in range(num_iterations):
        # Create random query
        query_embedding = create_test_embedding()
        
        # Measure search time
        start_time = time.perf_counter()
        results = vector_db.search_similar_faces(
            query_embedding=query_embedding,
            collection_name=collection_name,
            limit=search_limit,
            score_threshold=0.0,  # Get all results
            with_vectors=with_vectors
        )
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Verify results have vectors if requested
        if with_vectors and results:
            first_result = results[0]
            if 'vector' not in first_result:
                logger.warning(f"Iteration {i}: with_vectors=True but no vector in result!")
    
    return latencies


def calculate_statistics(latencies: List[float]) -> Dict[str, float]:
    """Calculate statistics from latency measurements."""
    if not latencies:
        return {}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        'mean': mean(latencies),
        'median': median(latencies),
        'std': stdev(latencies) if n > 1 else 0.0,
        'min': min(latencies),
        'max': max(latencies),
        'p50': sorted_latencies[int(n * 0.50)],
        'p95': sorted_latencies[int(n * 0.95)],
        'p99': sorted_latencies[int(n * 0.99)] if n > 10 else sorted_latencies[-1]
    }


def print_statistics(title: str, stats: Dict[str, float]) -> None:
    """Pretty print statistics."""
    print(f"\n{title}")
    print("=" * 60)
    print(f"  Mean:     {stats['mean']:.2f} ms")
    print(f"  Median:   {stats['median']:.2f} ms")
    print(f"  Std Dev:  {stats['std']:.2f} ms")
    print(f"  Min:      {stats['min']:.2f} ms")
    print(f"  Max:      {stats['max']:.2f} ms")
    print(f"  P50:      {stats['p50']:.2f} ms")
    print(f"  P95:      {stats['p95']:.2f} ms")
    print(f"  P99:      {stats['p99']:.2f} ms")


def main():
    """Run benchmark tests."""
    print("\n" + "=" * 60)
    print("BENCHMARK: with_vectors Performance Impact")
    print("=" * 60)
    
    # Initialize vector database
    try:
        vector_db = VectorDatabaseService(host="localhost", port=6333)
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        print("\n❌ ERROR: Could not connect to Qdrant.")
        print("   Make sure Qdrant is running on localhost:6333")
        return
    
    # Check collection stats
    try:
        missing_stats = vector_db.get_collection_stats("missing_persons")
        found_stats = vector_db.get_collection_stats("found_persons")
        
        print(f"\nCollection Statistics:")
        print(f"  Missing persons: {missing_stats['points_count']} records")
        print(f"  Found persons:   {found_stats['points_count']} records")
        
        if missing_stats['points_count'] == 0 and found_stats['points_count'] == 0:
            print("\n⚠️  WARNING: Collections are empty!")
            print("   Benchmark will still run but results may not be representative.")
            print("   Consider inserting test data first.")
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        print("\n❌ ERROR: Could not get collection statistics.")
        return
    
    # Configuration
    num_iterations = 50
    search_limit = 20
    collection = "missing_persons"
    
    print(f"\nBenchmark Configuration:")
    print(f"  Iterations:     {num_iterations}")
    print(f"  Search limit:   {search_limit} results")
    print(f"  Collection:     {collection}")
    
    # Warm-up
    print("\n⏳ Warming up...")
    _ = benchmark_search(vector_db, collection, num_iterations=5, search_limit=search_limit, with_vectors=False)
    _ = benchmark_search(vector_db, collection, num_iterations=5, search_limit=search_limit, with_vectors=True)
    
    # Benchmark WITHOUT vectors
    print(f"\n⏱️  Running benchmark WITHOUT vectors ({num_iterations} iterations)...")
    latencies_without_vectors = benchmark_search(
        vector_db, 
        collection, 
        num_iterations=num_iterations,
        search_limit=search_limit,
        with_vectors=False
    )
    stats_without = calculate_statistics(latencies_without_vectors)
    
    # Benchmark WITH vectors
    print(f"⏱️  Running benchmark WITH vectors ({num_iterations} iterations)...")
    latencies_with_vectors = benchmark_search(
        vector_db, 
        collection, 
        num_iterations=num_iterations,
        search_limit=search_limit,
        with_vectors=True
    )
    stats_with = calculate_statistics(latencies_with_vectors)
    
    # Print results
    print_statistics("📊 Results WITHOUT vectors", stats_without)
    print_statistics("📊 Results WITH vectors", stats_with)
    
    # Calculate overhead
    print("\n" + "=" * 60)
    print("Performance Impact Analysis")
    print("=" * 60)
    
    overhead_mean = stats_with['mean'] - stats_without['mean']
    overhead_pct = (overhead_mean / stats_without['mean']) * 100
    
    overhead_p95 = stats_with['p95'] - stats_without['p95']
    overhead_p95_pct = (overhead_p95 / stats_without['p95']) * 100
    
    print(f"\nOverhead (Mean):")
    print(f"  Absolute: +{overhead_mean:.2f} ms")
    print(f"  Relative: +{overhead_pct:.1f}%")
    
    print(f"\nOverhead (P95):")
    print(f"  Absolute: +{overhead_p95:.2f} ms")
    print(f"  Relative: +{overhead_p95_pct:.1f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("Verdict")
    print("=" * 60)
    
    if overhead_pct < 10:
        print("✅ NEGLIGIBLE OVERHEAD (<10%)")
        print(f"   with_vectors adds only {overhead_mean:.2f}ms on average.")
        print("   Recommendation: Safe to use with_vectors for multi-image aggregation.")
    elif overhead_pct < 30:
        print("⚠️  MODERATE OVERHEAD (10-30%)")
        print(f"   with_vectors adds {overhead_mean:.2f}ms on average.")
        print("   Recommendation: Acceptable for multi-image use cases.")
        print("   Consider limiting search results to reduce overhead.")
    else:
        print("❌ HIGH OVERHEAD (>30%)")
        print(f"   with_vectors adds {overhead_mean:.2f}ms on average.")
        print("   Recommendation: Consider optimization strategies:")
        print("   - Reduce search limit")
        print("   - Use separate retrieval step only when needed")
        print("   - Cache embeddings if reused")
    
    # Multi-image scenario estimate
    print("\n" + "=" * 60)
    print("Multi-Image Scenario Estimate")
    print("=" * 60)
    print("\nScenario: 5 query images × 20 target persons × 5 images each")
    print(f"  Total searches: 5")
    print(f"  Initial search limit: {search_limit * 10} (inflated for grouping)")
    
    estimated_latency_per_search = stats_with['mean']
    total_latency = estimated_latency_per_search * 5
    
    print(f"\nEstimated latencies:")
    print(f"  Per search (with vectors): {estimated_latency_per_search:.2f} ms")
    print(f"  Total search phase:        {total_latency:.2f} ms")
    print(f"  Aggregation phase:         ~5-10 ms (in-memory)")
    print(f"  TOTAL:                     ~{total_latency + 7:.0f} ms")
    
    if total_latency + 7 < 500:
        print(f"\n✅ MEETS TARGET (<500ms for multi-image search)")
    else:
        print(f"\n⚠️  EXCEEDS TARGET (>500ms)")
        print(f"   Consider optimizations or adjust expectations.")
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user.")
    except Exception as e:
        logger.exception("Benchmark failed with error")
        print(f"\n❌ ERROR: Benchmark failed: {e}")

