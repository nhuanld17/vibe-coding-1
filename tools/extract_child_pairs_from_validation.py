"""
Helper script to extract child pairs from existing validation_pairs.csv

This script can be used if your existing validation_pairs.csv contains
child images that can be identified by:
- Age information in metadata
- Folder names containing "child" or age < 18
- Manual filtering

Usage:
    python tools/extract_child_pairs_from_validation.py \
        --input datasets/validation_pairs.csv \
        --output datasets/validation_pairs_children.csv \
        --filter-by-age  # If age info is in CSV
"""

import sys
import csv
import argparse
from pathlib import Path
import re

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from loguru import logger

AGE_PATTERNS = [
    r'age[_\s]*(\d+)',
    r'(\d+)[_\s]*years?',
    r'(\d+)[_\s]*yo',
    r'child[_\s]*(\d+)',
]


def extract_age_from_path(path: str) -> int:
    """Try to extract age from file path."""
    for pattern in AGE_PATTERNS:
        match = re.search(pattern, path.lower())
        if match:
            try:
                age = int(match.group(1))
                if 0 <= age <= 18:
                    return age
            except ValueError:
                continue
    return None


def is_child_path(path: str) -> bool:
    """Check if path suggests a child image."""
    path_lower = path.lower()
    
    # Check for "child" in path
    if 'child' in path_lower:
        return True
    
    # Check for age < 18 in path
    age = extract_age_from_path(path)
    if age is not None and age < 18:
        return True
    
    return False


def filter_child_pairs(input_csv: Path, output_csv: Path) -> int:
    """
    Filter validation pairs to only include child pairs.
    
    Returns:
        Number of pairs written
    """
    child_pairs = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            img1 = row.get('image_path_1', '')
            img2 = row.get('image_path_2', '')
            
            # Check if both images are likely children
            is_child1 = is_child_path(img1)
            is_child2 = is_child_path(img2)
            
            if is_child1 or is_child2:
                # If either image is a child, include the pair
                # Add age fields if not present
                pair = row.copy()
                
                # Try to extract ages
                age1 = extract_age_from_path(img1)
                age2 = extract_age_from_path(img2)
                
                # Add age fields if not in CSV
                if 'age_1' not in pair:
                    pair['age_1'] = str(age1) if age1 is not None else ''
                if 'age_2' not in pair:
                    pair['age_2'] = str(age2) if age2 is not None else ''
                
                child_pairs.append(pair)
    
    # Write output
    if child_pairs:
        output_fieldnames = list(fieldnames)
        if 'age_1' not in output_fieldnames:
            output_fieldnames.insert(2, 'age_1')
        if 'age_2' not in output_fieldnames:
            output_fieldnames.insert(5, 'age_2')
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            writer.writerows(child_pairs)
        
        logger.info(f"Extracted {len(child_pairs)} child pairs to {output_csv}")
    else:
        logger.warning(f"No child pairs found in {input_csv}")
        logger.info("You may need to manually organize child images in datasets/children/")
    
    return len(child_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract child pairs from existing validation_pairs.csv"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/validation_pairs.csv',
        help='Input CSV file (default: datasets/validation_pairs.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/validation_pairs_children.csv',
        help='Output CSV file (default: datasets/validation_pairs_children.csv)'
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    print("=" * 80)
    print("EXTRACTING CHILD PAIRS FROM EXISTING VALIDATION DATA")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print("\nFiltering pairs where images contain 'child' or age < 18 in path...")
    
    count = filter_child_pairs(input_path, output_path)
    
    if count > 0:
        print(f"\n[OK] Extracted {count} child pairs")
        print(f"     Run: python tests/eval_threshold_sweep_children.py")
    else:
        print(f"\n[INFO] No child pairs found in existing data")
        print(f"       You need to:")
        print(f"       1. Organize child images in datasets/children/child_XXX/")
        print(f"       2. Run: python tools/build_validation_pairs_children.py")
    
    return 0


if __name__ == "__main__":
    exit(main())

