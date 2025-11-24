"""
Monitor child threshold evaluation progress in real-time.

Usage:
    python tools/monitor_evaluation.py
    # Or with auto-refresh:
    python tools/monitor_evaluation.py --watch
"""

import sys
import time
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

results_file = Path(__file__).parent.parent / "tests" / "threshold_sweep_children_results.csv"
validation_file = Path(__file__).parent.parent / "datasets" / "validation_pairs_children.csv"

def get_status():
    """Get current evaluation status."""
    status = {
        'validation_exists': validation_file.exists(),
        'validation_pairs': 0,
        'results_exists': results_file.exists(),
        'results_lines': 0,
        'last_threshold': None,
        'progress_pct': 0.0
    }
    
    if status['validation_exists']:
        with open(validation_file, 'r', encoding='utf-8') as f:
            status['validation_pairs'] = sum(1 for _ in f) - 1  # Subtract header
    
    if status['results_exists']:
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            status['results_lines'] = len(lines) - 1  # Subtract header
            if len(lines) > 1:
                # Parse last line to get threshold
                last_line = lines[-1].strip()
                if last_line:
                    parts = last_line.split(',')
                    if len(parts) > 0:
                        try:
                            status['last_threshold'] = float(parts[0])
                        except:
                            pass
        
        # Calculate progress (expected ~51 thresholds for 0.30-0.80 step 0.01)
        expected = int((0.80 - 0.30) / 0.01) + 1
        if expected > 0:
            status['progress_pct'] = min(100.0, (status['results_lines'] / expected) * 100)
    
    return status

def print_status(status):
    """Print formatted status."""
    print("\n" + "=" * 80)
    print("CHILD THRESHOLD EVALUATION - LIVE STATUS")
    print("=" * 80)
    
    if status['validation_exists']:
        print(f"\n✅ Validation Data: {status['validation_pairs']} pairs")
    else:
        print(f"\n❌ Validation data not found")
    
    if status['results_exists']:
        print(f"\n✅ Results File: {status['results_lines']} thresholds evaluated")
        if status['last_threshold']:
            print(f"   Last threshold: {status['last_threshold']:.2f}")
        print(f"   Progress: {status['progress_pct']:.1f}%")
        
        if status['progress_pct'] >= 100:
            print(f"\n🎉 EVALUATION COMPLETE!")
            print(f"   Check results: tests/threshold_sweep_children_results.csv")
        else:
            print(f"\n⏳ Evaluation in progress...")
    else:
        print(f"\n⏳ Results file not created yet")
        print(f"   Evaluation is initializing (loading pairs, computing embeddings)...")
    
    print("=" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor child threshold evaluation")
    parser.add_argument('--watch', action='store_true', help='Auto-refresh every 10 seconds')
    args = parser.parse_args()
    
    if args.watch:
        print("Monitoring evaluation (press Ctrl+C to stop)...")
        try:
            while True:
                status = get_status()
                print_status(status)
                if status['progress_pct'] >= 100:
                    print("\n✅ Evaluation complete! Exiting...")
                    break
                print("\nRefreshing in 10 seconds...")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        status = get_status()
        print_status(status)

if __name__ == "__main__":
    main()

