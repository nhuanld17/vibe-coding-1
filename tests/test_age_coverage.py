"""
Test script to check age coverage in validation data (0-18 years old).

This script analyzes validation_pairs_children.csv to verify that
all ages from 0 to 18 are covered in the test data.
"""

import sys
from pathlib import Path
from collections import Counter

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import csv

def analyze_age_coverage():
    """Analyze age coverage in validation_pairs_children.csv"""
    csv_path = Path(__file__).parent.parent / "datasets" / "validation_pairs_children.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] Validation file not found: {csv_path}")
        return
    
    ages = []
    age_pairs = []
    
    print("=" * 80)
    print("AGE COVERAGE ANALYSIS (0-18 years)")
    print("=" * 80)
    print(f"\nReading: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age1_str = row.get('age_1', '').strip()
            age2_str = row.get('age_2', '').strip()
            
            # Extract age1
            if age1_str:
                try:
                    age1 = int(age1_str)
                    if 0 <= age1 <= 18:
                        ages.append(age1)
                except (ValueError, TypeError):
                    pass
            
            # Extract age2
            if age2_str:
                try:
                    age2 = int(age2_str)
                    if 0 <= age2 <= 18:
                        ages.append(age2)
                except (ValueError, TypeError):
                    pass
            
            # Store pairs for analysis
            if age1_str and age2_str:
                try:
                    a1 = int(age1_str) if age1_str else None
                    a2 = int(age2_str) if age2_str else None
                    if a1 is not None and a2 is not None and 0 <= a1 <= 18 and 0 <= a2 <= 18:
                        age_pairs.append((a1, a2))
                except (ValueError, TypeError):
                    pass
    
    age_counts = Counter(ages)
    
    print(f"\nTotal age entries (0-18): {len(ages)}")
    print(f"Unique ages found: {len(age_counts)}")
    print(f"Age pairs analyzed: {len(age_pairs)}")
    
    # Check coverage for ages 0-18
    print("\n" + "=" * 80)
    print("AGE DISTRIBUTION (0-18)")
    print("=" * 80)
    
    all_ages = sorted(age_counts.keys())
    missing_ages = [age for age in range(19) if age not in age_counts]
    
    if all_ages:
        print(f"\nAges found in data:")
        for age in sorted(all_ages):
            count = age_counts[age]
            bar = "█" * min(count // 10, 50)  # Visual bar
            print(f"  Age {age:2d}: {count:5d} entries {bar}")
    
    if missing_ages:
        print(f"\n⚠️  MISSING AGES (0-18): {missing_ages}")
        print(f"   Total missing: {len(missing_ages)} ages")
    else:
        print(f"\n✅ ALL AGES 0-18 ARE COVERED!")
        print(f"   Perfect coverage: {len(age_counts)} unique ages")
    
    # Analyze age pairs
    print("\n" + "=" * 80)
    print("AGE PAIR ANALYSIS")
    print("=" * 80)
    
    same_child_pairs = [p for p in age_pairs if p[0] == p[1]]
    diff_age_pairs = [p for p in age_pairs if p[0] != p[1]]
    
    print(f"\nSame-child pairs (same age): {len(same_child_pairs)}")
    print(f"Different-age pairs: {len(diff_age_pairs)}")
    
    # Age gap distribution
    age_gaps = Counter()
    for a1, a2 in diff_age_pairs:
        gap = abs(a1 - a2)
        age_gaps[gap] += 1
    
    if age_gaps:
        print(f"\nAge gap distribution (different-age pairs):")
        for gap in sorted(age_gaps.keys())[:10]:  # Top 10 gaps
            print(f"  Gap {gap:2d} years: {age_gaps[gap]:5d} pairs")
    
    # Check if we have pairs covering all age ranges
    print("\n" + "=" * 80)
    print("AGE RANGE COVERAGE")
    print("=" * 80)
    
    age_ranges = {
        "Infant (0-2)": (0, 2),
        "Toddler (3-5)": (3, 5),
        "Child (6-9)": (6, 9),
        "Pre-teen (10-12)": (10, 12),
        "Teen (13-15)": (13, 15),
        "Young adult (16-18)": (16, 18)
    }
    
    for range_name, (min_age, max_age) in age_ranges.items():
        range_ages = [a for a in ages if min_age <= a <= max_age]
        if range_ages:
            print(f"  ✅ {range_name}: {len(range_ages)} entries")
        else:
            print(f"  ⚠️  {range_name}: NO DATA")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    coverage_percent = (len(age_counts) / 19) * 100
    print(f"\nCoverage: {len(age_counts)}/19 ages ({coverage_percent:.1f}%)")
    
    if missing_ages:
        print(f"⚠️  Missing ages: {missing_ages}")
        print(f"\nRecommendation: Add test data for missing ages to ensure")
        print(f"  comprehensive testing of child detection logic (age < 18).")
    else:
        print(f"✅ Perfect coverage! All ages 0-18 are present in test data.")
        print(f"  This ensures comprehensive testing of child detection logic.")

if __name__ == "__main__":
    analyze_age_coverage()

