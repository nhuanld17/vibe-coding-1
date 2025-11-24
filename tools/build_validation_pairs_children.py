"""
Child-Focused Validation Data Toolkit - Build validation_pairs_children.csv

This script is specifically designed for building validation pairs for CHILDREN ONLY.
Children's faces are less distinctive than adults, requiring separate evaluation.

HOW TO USE:
-----------
1. Organize your child images in folders:

   datasets/children/
     child_001/
       img1.jpg
       img2.jpg
       ...
     child_002/
       img1.jpg
       ...

   Or use multiple root folders:
     datasets/children/
     datasets/child_val/

2. Run the script:
   python tools/build_validation_pairs_children.py
   
   Or with custom config:
   python tools/build_validation_pairs_children.py --roots datasets/children --max-same 50

3. The script will:
   - Scan all child folders
   - Generate same-child pairs (within each child folder)
   - Generate different-child pairs (cross-child, balanced)
   - Output datasets/validation_pairs_children.csv with age information

OUTPUT FORMAT:
--------------
CSV with columns: person_id_1,image_path_1,age_1,person_id_2,image_path_2,age_2,label
- label=1 for same child
- label=0 for different children
- age_1, age_2: Age or age group (if available from folder name or metadata)
- image paths are relative to repo root

NOTE: Age fields are optional. If age cannot be determined, they will be empty.
"""

import sys
import os
import csv
import random
import argparse
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from loguru import logger

# ============================================================================
# CONFIGURATION (can be overridden via CLI)
# ============================================================================

# Default root folders to scan for children (relative to repo root)
DEFAULT_ROOTS = [
    "datasets/children",
    "datasets/child_val"
]

# Maximum same-child pairs per child (to avoid explosion)
DEFAULT_MAX_SAME_PAIRS_PER_CHILD = 50

# Maximum different-child pairs total (will be balanced with same-child pairs)
DEFAULT_MAX_DIFF_PAIRS = None  # None = match same-child count

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# Random seed for reproducibility
RANDOM_SEED = 42

# Age extraction patterns (try to extract age from folder name or filename)
AGE_PATTERNS = [
    r'age[_\s]*(\d+)',  # age_5, age 5
    r'(\d+)[_\s]*years?',  # 5 years, 5_years
    r'(\d+)[_\s]*yo',  # 5yo, 5_yo
    r'child[_\s]*(\d+)',  # child_5
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_age_from_name(name: str) -> Optional[int]:
    """
    Try to extract age from folder name or filename.
    
    Args:
        name: Folder name or filename
        
    Returns:
        Age as integer if found, None otherwise
    """
    name_lower = name.lower()
    
    for pattern in AGE_PATTERNS:
        match = re.search(pattern, name_lower)
        if match:
            try:
                age = int(match.group(1))
                if 0 <= age <= 18:  # Reasonable child age range
                    return age
            except ValueError:
                continue
    
    return None


def scan_child_folders(root_paths: List[str], repo_root: Path) -> Dict[str, Dict]:
    """
    Scan root folders for child directories and collect image paths with age info.
    
    Args:
        root_paths: List of root folder paths (relative to repo_root)
        repo_root: Repository root path
        
    Returns:
        Dict mapping child_id -> {
            'images': [list of image paths],
            'age': age (int or None)
        }
    """
    child_data = {}
    
    for root_path in root_paths:
        root_abs = repo_root / root_path
        
        if not root_abs.exists():
            logger.warning(f"Root folder not found: {root_abs}")
            continue
        
        logger.info(f"Scanning: {root_abs}")
        
        # Look for child_* directories or any directory
        for child_dir in root_abs.iterdir():
            if not child_dir.is_dir():
                continue
            
            child_id = child_dir.name
            
            # Try to extract age from folder name
            age = extract_age_from_name(child_id)
            
            # Collect images in this child folder
            images = []
            for img_file in child_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    # Store relative path from repo root
                    rel_path = img_file.relative_to(repo_root)
                    images.append(str(rel_path).replace('\\', '/'))  # Normalize to forward slashes
                    
                    # Also try to extract age from filename if not found
                    if age is None:
                        age = extract_age_from_name(img_file.stem)
            
            if images:
                # Use full path as child_id if we have multiple roots with same child names
                full_child_id = f"{root_path}/{child_id}"
                if full_child_id in child_data:
                    # Merge with existing
                    child_data[full_child_id]['images'].extend(images)
                    # Update age if we found one
                    if age is not None:
                        child_data[full_child_id]['age'] = age
                else:
                    child_data[full_child_id] = {
                        'images': images,
                        'age': age
                    }
                
                logger.debug(f"  Found {len(images)} images in {child_id} (age: {age if age else 'unknown'})")
    
    return child_data


def generate_same_child_pairs(
    child_data: Dict[str, Dict],
    max_pairs_per_child: int
) -> List[Dict[str, str]]:
    """
    Generate same-child pairs (all combinations within each child folder).
    
    Args:
        child_data: Dict mapping child_id -> {'images': [...], 'age': ...}
        max_pairs_per_child: Maximum pairs to generate per child
        
    Returns:
        List of pair dicts with keys: person_id_1, image_path_1, age_1, person_id_2, image_path_2, age_2, label
    """
    pairs = []
    
    for child_id, data in child_data.items():
        images = data['images']
        age = data.get('age')
        
        if len(images) < 2:
            logger.debug(f"Skipping {child_id}: need at least 2 images, found {len(images)}")
            continue
        
        # Generate all combinations
        combinations = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                combinations.append((images[i], images[j]))
        
        # Limit if needed
        if max_pairs_per_child and len(combinations) > max_pairs_per_child:
            random.shuffle(combinations)
            combinations = combinations[:max_pairs_per_child]
        
        # Add pairs
        for img1, img2 in combinations:
            pairs.append({
                'person_id_1': child_id,
                'image_path_1': img1,
                'age_1': str(age) if age is not None else '',
                'person_id_2': child_id,
                'image_path_2': img2,
                'age_2': str(age) if age is not None else '',
                'label': 1
            })
    
    logger.info(f"Generated {len(pairs)} same-child pairs")
    return pairs


def generate_different_child_pairs(
    child_data: Dict[str, Dict],
    target_count: int
) -> List[Dict[str, str]]:
    """
    Generate different-child pairs (randomly sampled cross-child pairs).
    
    Args:
        child_data: Dict mapping child_id -> {'images': [...], 'age': ...}
        target_count: Target number of pairs to generate (None = match same-child count)
        
    Returns:
        List of pair dicts with keys: person_id_1, image_path_1, age_1, person_id_2, image_path_2, age_2, label
    """
    pairs = []
    child_ids = list(child_data.keys())
    
    if len(child_ids) < 2:
        logger.warning("Need at least 2 children to generate different-child pairs")
        return pairs
    
    # Generate candidate pairs
    candidates = []
    for i in range(len(child_ids)):
        for j in range(i + 1, len(child_ids)):
            child1_id = child_ids[i]
            child2_id = child_ids[j]
            
            data1 = child_data[child1_id]
            data2 = child_data[child2_id]
            
            images1 = data1['images']
            images2 = data2['images']
            age1 = data1.get('age')
            age2 = data2.get('age')
            
            # Sample one image from each child
            for img1 in images1:
                for img2 in images2:
                    candidates.append((child1_id, img1, age1, child2_id, img2, age2))
    
    # Sample target_count pairs
    if target_count is None:
        # Match same-child count (will be set later)
        target_count = len(candidates)
    
    if len(candidates) > target_count:
        random.shuffle(candidates)
        candidates = candidates[:target_count]
    
    # Create pairs
    for child1_id, img1, age1, child2_id, img2, age2 in candidates:
        pairs.append({
            'person_id_1': child1_id,
            'image_path_1': img1,
            'age_1': str(age1) if age1 is not None else '',
            'person_id_2': child2_id,
            'image_path_2': img2,
            'age_2': str(age2) if age2 is not None else '',
            'label': 0
        })
    
    logger.info(f"Generated {len(pairs)} different-child pairs")
    return pairs


def write_validation_csv(pairs: List[Dict[str, str]], output_path: Path):
    """
    Write validation pairs to CSV file with age information.
    
    Args:
        pairs: List of pair dicts
        output_path: Output CSV file path
    """
    # Shuffle pairs for randomness
    random.shuffle(pairs)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['person_id_1', 'image_path_1', 'age_1', 'person_id_2', 'image_path_2', 'age_2', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    
    logger.info(f"Wrote {len(pairs)} pairs to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build validation_pairs_children.csv from organized child image folders"
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        default=DEFAULT_ROOTS,
        help=f"Root folders to scan (default: {DEFAULT_ROOTS})"
    )
    parser.add_argument(
        '--max-same',
        type=int,
        default=DEFAULT_MAX_SAME_PAIRS_PER_CHILD,
        help=f"Maximum same-child pairs per child (default: {DEFAULT_MAX_SAME_PAIRS_PER_CHILD})"
    )
    parser.add_argument(
        '--max-diff',
        type=int,
        default=None,
        help="Maximum different-child pairs (default: match same-child count)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV path (default: datasets/validation_pairs_children.csv)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    print("=" * 80)
    print("CHILD-FOCUSED VALIDATION DATA TOOLKIT")
    print("Building validation_pairs_children.csv")
    print("=" * 80)
    
    # Get repo root
    repo_root = Path(__file__).parent.parent
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / "datasets" / "validation_pairs_children.csv"
    
    # Scan child folders
    print(f"\nScanning root folders: {args.roots}")
    child_data = scan_child_folders(args.roots, repo_root)
    
    if not child_data:
        print("[ERROR] No child folders found. Check your root paths.")
        return 1
    
    print(f"\nFound {len(child_data)} children with images:")
    total_images = sum(len(data['images']) for data in child_data.values())
    children_with_age = sum(1 for data in child_data.values() if data.get('age') is not None)
    print(f"  Total images: {total_images}")
    print(f"  Average images per child: {total_images / len(child_data):.1f}")
    print(f"  Children with age info: {children_with_age}/{len(child_data)}")
    
    # Generate same-child pairs
    print(f"\nGenerating same-child pairs (max {args.max_same} per child)...")
    same_pairs = generate_same_child_pairs(child_data, args.max_same)
    
    # Generate different-child pairs
    target_diff = args.max_diff if args.max_diff is not None else len(same_pairs)
    print(f"\nGenerating different-child pairs (target: {target_diff})...")
    diff_pairs = generate_different_child_pairs(child_data, target_diff)
    
    # Combine pairs
    all_pairs = same_pairs + diff_pairs
    
    print(f"\nSummary:")
    print(f"  Same-child pairs: {len(same_pairs)}")
    print(f"  Different-child pairs: {len(diff_pairs)}")
    print(f"  Total pairs: {len(all_pairs)}")
    
    # Write CSV
    print(f"\nWriting to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_validation_csv(all_pairs, output_path)
    
    print(f"\n[OK] Child validation pairs CSV created successfully!")
    print(f"     Run eval_threshold_sweep_children.py to evaluate child-specific thresholds.")
    
    return 0


if __name__ == "__main__":
    exit(main())

