"""
Quick script to check the status of child threshold evaluation.

Usage:
    python tools/check_evaluation_status.py
"""

import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

results_file = Path(__file__).parent.parent / "tests" / "threshold_sweep_children_results.csv"
validation_file = Path(__file__).parent.parent / "datasets" / "validation_pairs_children.csv"

print("=" * 80)
print("CHILD THRESHOLD EVALUATION STATUS")
print("=" * 80)

# Check validation data
if validation_file.exists():
    with open(validation_file, 'r', encoding='utf-8') as f:
        lines = sum(1 for _ in f)
    print(f"\n✅ Validation data: {validation_file}")
    print(f"   Total pairs: {lines - 1}")  # Subtract header
else:
    print(f"\n❌ Validation data not found: {validation_file}")
    print("   Run: python tools/extract_child_pairs_from_validation.py")

# Check results
if results_file.exists():
    with open(results_file, 'r', encoding='utf-8') as f:
        lines = sum(1 for _ in f)
    
    print(f"\n✅ Results file: {results_file}")
    print(f"   Total thresholds evaluated: {lines - 1}")  # Subtract header
    
    if lines > 1:
        # Read last few lines to see progress
        with open(results_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            print(f"\n   Last evaluated threshold:")
            print(f"   {all_lines[-1].strip()}")
        
        # Check if evaluation is complete (should have ~50 thresholds for 0.30-0.80 step 0.01)
        expected_thresholds = int((0.80 - 0.30) / 0.01) + 1
        if lines - 1 >= expected_thresholds:
            print(f"\n   ✅ Evaluation appears COMPLETE!")
            print(f"   Expected ~{expected_thresholds} thresholds, found {lines - 1}")
        else:
            print(f"\n   ⏳ Evaluation IN PROGRESS...")
            print(f"   Expected ~{expected_thresholds} thresholds, found {lines - 1}")
            print(f"   Progress: {((lines - 1) / expected_thresholds * 100):.1f}%")
    else:
        print(f"\n   ⏳ Results file exists but only has header")
        print(f"   Evaluation is starting...")
else:
    print(f"\n⏳ Results file not found: {results_file}")
    print("   Evaluation has not started or is still initializing")
    print("\n   To start evaluation:")
    print("   python tests/eval_threshold_sweep_children.py")

print("\n" + "=" * 80)

