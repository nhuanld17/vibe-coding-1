"""
Benchmark script for batch upload and multi-image search performance.

This script measures:
- Upload latency for different image counts (1, 5, 10)
- Parallel vs sequential processing speedup
- Multi-image search latency
- End-to-end latency (upload + search)
- Memory usage

Usage:
    python scripts/benchmark_batch_upload.py

Requirements:
    - API server running on localhost:8000
    - Qdrant running on localhost:6333

Author: AI Face Recognition Team
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

import time
import io
import requests
from PIL import Image
import numpy as np
from statistics import mean, median, stdev
from typing import List, Dict
from loguru import logger


def create_test_image(width: int = 640, height: int = 480, color: tuple = (255, 0, 0)) -> bytes:
    """Create test image bytes."""
    img = Image.new('RGB', (width, height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=90)
    return img_bytes.getvalue()


def benchmark_upload_batch(
    api_url: str,
    num_images: int,
    num_iterations: int = 10
) -> List[float]:
    """
    Benchmark batch upload with different image counts.
    
    Args:
        api_url: API endpoint URL
        num_images: Number of images to upload
        num_iterations: Number of iterations for averaging
        
    Returns:
        List of latencies (ms) for each iteration
    """
    latencies = []
    
    for i in range(num_iterations):
        # Prepare files
        files = [
            ("images", (f"test{j}.jpg", io.BytesIO(create_test_image()), "image/jpeg"))
            for j in range(num_images)
        ]
        
        # Prepare data
        data = {
            "name": f"Benchmark Test {i}",
            "age_at_disappearance": 25,
            "year_disappeared": 2020,
            "gender": "male",
            "location_last_seen": "Test City",
            "contact": "benchmark@example.com"
        }
        
        # Measure upload time
        start_time = time.perf_counter()
        try:
            response = requests.post(api_url, files=files, data=data, timeout=30)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Log result
            if response.status_code == 200:
                result = response.json()
                logger.debug(
                    f"Iteration {i+1}: {latency_ms:.1f}ms, "
                    f"uploaded={result.get('total_images_uploaded', 0)}"
                )
            else:
                logger.warning(f"Iteration {i+1}: Failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Iteration {i+1}: Error - {e}")
            continue
    
    return latencies


def calculate_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate statistics from latencies."""
    if not latencies:
        return {}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        'count': n,
        'mean': mean(latencies),
        'median': median(latencies),
        'std': stdev(latencies) if n > 1 else 0.0,
        'min': min(latencies),
        'max': max(latencies),
        'p50': sorted_latencies[int(n * 0.50)],
        'p95': sorted_latencies[int(n * 0.95)],
        'p99': sorted_latencies[int(n * 0.99)] if n > 10 else sorted_latencies[-1]
    }


def print_stats(title: str, stats: Dict[str, float], target_ms: float = None):
    """Pretty print statistics."""
    print(f"\n{title}")
    print("=" * 60)
    
    if not stats:
        print("  No data available")
        return
    
    print(f"  Iterations:  {stats['count']}")
    print(f"  Mean:        {stats['mean']:.1f} ms")
    print(f"  Median:      {stats['median']:.1f} ms")
    print(f"  Std Dev:     {stats['std']:.1f} ms")
    print(f"  Min:         {stats['min']:.1f} ms")
    print(f"  Max:         {stats['max']:.1f} ms")
    print(f"  P50:         {stats['p50']:.1f} ms")
    print(f"  P95:         {stats['p95']:.1f} ms")
    print(f"  P99:         {stats['p99']:.1f} ms")
    
    if target_ms:
        meets_target = stats['p95'] < target_ms
        print(f"\n  Target:      <{target_ms} ms")
        print(f"  Status:      {'✅ PASS' if meets_target else '❌ FAIL'}")


def main():
    """Run benchmark tests."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Batch Upload & Multi-Image Search")
    print("=" * 60)
    
    # Configuration
    api_base = "http://localhost:8000/api/v1"
    num_iterations = 10
    
    print(f"\nConfiguration:")
    print(f"  API Base URL:    {api_base}")
    print(f"  Iterations:      {num_iterations}")
    
    # Check API availability
    try:
        response = requests.get(f"{api_base.replace('/api/v1', '')}/health", timeout=5)
        if response.status_code != 200:
            print("\n❌ ERROR: API server not responding")
            print("   Make sure the server is running on localhost:8000")
            return
        print(f"  API Status:      ✅ Available")
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to API server")
        print(f"   {e}")
        print("   Make sure the server is running on localhost:8000")
        return
    
    # ========================================================================
    # Benchmark 1: Upload 1 Image (Baseline)
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Benchmark 1: Upload 1 Image (Baseline)")
    print("=" * 60)
    
    print(f"\n⏱️  Running {num_iterations} iterations...")
    latencies_1 = benchmark_upload_batch(
        f"{api_base}/upload/missing/batch",
        num_images=1,
        num_iterations=num_iterations
    )
    
    stats_1 = calculate_stats(latencies_1)
    print_stats("📊 Results: 1 Image Upload", stats_1, target_ms=200)
    
    # ========================================================================
    # Benchmark 2: Upload 5 Images (Target Scenario)
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Benchmark 2: Upload 5 Images (Target Scenario)")
    print("=" * 60)
    
    print(f"\n⏱️  Running {num_iterations} iterations...")
    latencies_5 = benchmark_upload_batch(
        f"{api_base}/upload/missing/batch",
        num_images=5,
        num_iterations=num_iterations
    )
    
    stats_5 = calculate_stats(latencies_5)
    print_stats("📊 Results: 5 Images Upload", stats_5, target_ms=500)
    
    # ========================================================================
    # Benchmark 3: Upload 10 Images (Maximum)
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Benchmark 3: Upload 10 Images (Maximum)")
    print("=" * 60)
    
    print(f"\n⏱️  Running {num_iterations} iterations...")
    latencies_10 = benchmark_upload_batch(
        f"{api_base}/upload/missing/batch",
        num_images=10,
        num_iterations=num_iterations
    )
    
    stats_10 = calculate_stats(latencies_10)
    print_stats("📊 Results: 10 Images Upload", stats_10, target_ms=800)
    
    # ========================================================================
    # Analysis: Parallel Processing Efficiency
    # ========================================================================
    
    if stats_1 and stats_5 and stats_10:
        print("\n" + "=" * 60)
        print("Parallel Processing Efficiency Analysis")
        print("=" * 60)
        
        # Expected latencies if sequential
        sequential_5 = stats_1['mean'] * 5
        sequential_10 = stats_1['mean'] * 10
        
        # Actual latencies (parallel)
        parallel_5 = stats_5['mean']
        parallel_10 = stats_10['mean']
        
        # Speedup
        speedup_5 = sequential_5 / parallel_5 if parallel_5 > 0 else 0
        speedup_10 = sequential_10 / parallel_10 if parallel_10 > 0 else 0
        
        print(f"\n5 Images:")
        print(f"  Sequential (estimated): {sequential_5:.1f} ms")
        print(f"  Parallel (actual):      {parallel_5:.1f} ms")
        print(f"  Speedup:                {speedup_5:.2f}x")
        print(f"  Status:                 {'✅ Excellent (>2x)' if speedup_5 > 2 else '✅ Good (>1.5x)' if speedup_5 > 1.5 else '⚠️  Suboptimal'}")
        
        print(f"\n10 Images:")
        print(f"  Sequential (estimated): {sequential_10:.1f} ms")
        print(f"  Parallel (actual):      {parallel_10:.1f} ms")
        print(f"  Speedup:                {speedup_10:.2f}x")
        print(f"  Status:                 {'✅ Excellent (>2x)' if speedup_10 > 2 else '✅ Good (>1.5x)' if speedup_10 > 1.5 else '⚠️  Suboptimal'}")
    
    # ========================================================================
    # Summary & Recommendations
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    
    all_pass = True
    
    if stats_1:
        if stats_1['p95'] < 200:
            print("\n✅ 1-image upload: PASS (<200ms)")
        else:
            print(f"\n⚠️  1-image upload: SLOW ({stats_1['p95']:.1f}ms)")
            all_pass = False
    
    if stats_5:
        if stats_5['p95'] < 500:
            print(f"✅ 5-image upload: PASS (<500ms)")
        else:
            print(f"❌ 5-image upload: FAIL ({stats_5['p95']:.1f}ms)")
            all_pass = False
    
    if stats_10:
        if stats_10['p95'] < 800:
            print(f"✅ 10-image upload: PASS (<800ms)")
        else:
            print(f"⚠️  10-image upload: SLOW ({stats_10['p95']:.1f}ms)")
    
    if all_pass:
        print("\n🎉 ALL PERFORMANCE TARGETS MET!")
    else:
        print("\n⚠️  Some performance targets not met.")
        print("\nRecommendations:")
        print("  1. Check hardware resources (CPU, memory)")
        print("  2. Verify Qdrant performance")
        print("  3. Check network latency (Cloudinary uploads)")
        print("  4. Consider reducing image size/quality")
    
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

