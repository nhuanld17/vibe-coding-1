"""
Validation Data Toolkit - Build validation_pairs.csv from organized image folders

HOW TO USE:
-----------
1. Organize your validation images in folders:
   
   datasets/real_val/
     person_001/
       img1.jpg
       img2.jpg
       ...
     person_002/
       img1.jpg
       ...
   
   Or use multiple root folders:
     datasets/real_val/
     datasets/FGNET_organized/

2. Run the script:
   python tools/build_validation_pairs.py
   
   Or with custom config:
   python tools/build_validation_pairs.py --roots datasets/real_val datasets/FGNET_organized --max-same 50

3. The script will:
   - Scan all person folders
   - Generate same-person pairs (within each person folder)
   - Generate different-person pairs (cross-person, balanced)
   - Output datasets/validation_pairs.csv in the format expected by eval_threshold_sweep.py

OUTPUT FORMAT:
--------------
CSV with columns: person_id_1,image_path_1,person_id_2,image_path_2,label
- label=1 for same person
- label=0 for different person
- image paths are relative to repo root
"""

import sys
import os
import csv
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

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

# Default root folders to scan (relative to repo root)
DEFAULT_ROOTS = [
    "datasets/real_val",
    "datasets/FGNET_organized"
]

# Maximum same-person pairs per person (to avoid explosion)
DEFAULT_MAX_SAME_PAIRS_PER_PERSON = 50

# Maximum different-person pairs total (will be balanced with same-person pairs)
DEFAULT_MAX_DIFF_PAIRS = None  # None = match same-person count

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def scan_person_folders(root_paths: List[str], repo_root: Path) -> Dict[str, List[str]]:
    """
    Scan root folders for person directories and collect image paths.
    
    Args:
        root_paths: List of root folder paths (relative to repo_root)
        repo_root: Repository root path
        
    Returns:
        Dict mapping person_id -> list of image paths (relative to repo_root)
    """
    person_images = {}
    
    for root_path in root_paths:
        root_abs = repo_root / root_path
        
        if not root_abs.exists():
            logger.warning(f"Root folder not found: {root_abs}")
            continue
        
        logger.info(f"Scanning: {root_abs}")
        
        # Look for person_* directories
        for person_dir in root_abs.iterdir():
            if not person_dir.is_dir():
                continue
            
            # Check if it looks like a person directory
            person_id = person_dir.name
            if not (person_id.startswith('person_') or person_id.isdigit() or person_id.startswith('P')):
                # Allow any directory name as person_id
                pass
            
            # Collect images in this person folder
            images = []
            for img_file in person_dir.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    # Store relative path from repo root
                    rel_path = img_file.relative_to(repo_root)
                    images.append(str(rel_path).replace('\\', '/'))  # Normalize to forward slashes
            
            if images:
                # Use full path as person_id if we have multiple roots with same person names
                full_person_id = f"{root_path}/{person_id}"
                if full_person_id in person_images:
                    # Merge with existing
                    person_images[full_person_id].extend(images)
                else:
                    person_images[full_person_id] = images
                
                logger.debug(f"  Found {len(images)} images in {person_id}")
    
    return person_images


def generate_same_person_pairs(
    person_images: Dict[str, List[str]],
    max_pairs_per_person: int
) -> List[Dict[str, str]]:
    """
    Generate same-person pairs (all combinations within each person folder).
    
    Args:
        person_images: Dict mapping person_id -> list of image paths
        max_pairs_per_person: Maximum pairs to generate per person
        
    Returns:
        List of pair dicts with keys: person_id_1, image_path_1, person_id_2, image_path_2, label
    """
    pairs = []
    
    for person_id, images in person_images.items():
        if len(images) < 2:
            logger.debug(f"Skipping {person_id}: need at least 2 images, found {len(images)}")
            continue
        
        # Generate all combinations
        combinations = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                combinations.append((images[i], images[j]))
        
        # Limit if needed
        if max_pairs_per_person and len(combinations) > max_pairs_per_person:
            random.shuffle(combinations)
            combinations = combinations[:max_pairs_per_person]
        
        # Add pairs
        for img1, img2 in combinations:
            pairs.append({
                'person_id_1': person_id,
                'image_path_1': img1,
                'person_id_2': person_id,
                'image_path_2': img2,
                'label': 1
            })
    
    logger.info(f"Generated {len(pairs)} same-person pairs")
    return pairs


def generate_different_person_pairs(
    person_images: Dict[str, List[str]],
    target_count: int
) -> List[Dict[str, str]]:
    """
    Generate different-person pairs (randomly sampled cross-person pairs).
    
    Args:
        person_images: Dict mapping person_id -> list of image paths
        target_count: Target number of pairs to generate (None = match same-person count)
        
    Returns:
        List of pair dicts with keys: person_id_1, image_path_1, person_id_2, image_path_2, label
    """
    pairs = []
    person_ids = list(person_images.keys())
    
    if len(person_ids) < 2:
        logger.warning("Need at least 2 persons to generate different-person pairs")
        return pairs
    
    # Generate candidate pairs
    candidates = []
    for i in range(len(person_ids)):
        for j in range(i + 1, len(person_ids)):
            person1 = person_ids[i]
            person2 = person_ids[j]
            
            images1 = person_images[person1]
            images2 = person_images[person2]
            
            # Sample one image from each person
            for img1 in images1:
                for img2 in images2:
                    candidates.append((person1, img1, person2, img2))
    
    # Sample target_count pairs
    if target_count is None:
        # Match same-person count (will be set later)
        target_count = len(candidates)
    
    if len(candidates) > target_count:
        random.shuffle(candidates)
        candidates = candidates[:target_count]
    
    # Create pairs
    for person1, img1, person2, img2 in candidates:
        pairs.append({
            'person_id_1': person1,
            'image_path_1': img1,
            'person_id_2': person2,
            'image_path_2': img2,
            'label': 0
        })
    
    logger.info(f"Generated {len(pairs)} different-person pairs")
    return pairs


def write_validation_csv(pairs: List[Dict[str, str]], output_path: Path):
    """
    Write validation pairs to CSV file.
    
    Args:
        pairs: List of pair dicts
        output_path: Output CSV file path
    """
    # Shuffle pairs for randomness
    random.shuffle(pairs)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['person_id_1', 'image_path_1', 'person_id_2', 'image_path_2', 'label']
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
        description="Build validation_pairs.csv from organized image folders"
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
        default=DEFAULT_MAX_SAME_PAIRS_PER_PERSON,
        help=f"Maximum same-person pairs per person (default: {DEFAULT_MAX_SAME_PAIRS_PER_PERSON})"
    )
    parser.add_argument(
        '--max-diff',
        type=int,
        default=None,
        help="Maximum different-person pairs (default: match same-person count)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV path (default: datasets/validation_pairs.csv)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(RANDOM_SEED)
    
    print("=" * 80)
    print("VALIDATION DATA TOOLKIT - Building validation_pairs.csv")
    print("=" * 80)
    
    # Get repo root
    repo_root = Path(__file__).parent.parent
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / "datasets" / "validation_pairs.csv"
    
    # Scan person folders
    print(f"\nScanning root folders: {args.roots}")
    person_images = scan_person_folders(args.roots, repo_root)
    
    if not person_images:
        print("[ERROR] No person folders found. Check your root paths.")
        return 1
    
    print(f"\nFound {len(person_images)} persons with images:")
    total_images = sum(len(imgs) for imgs in person_images.values())
    print(f"  Total images: {total_images}")
    print(f"  Average images per person: {total_images / len(person_images):.1f}")
    
    # Generate same-person pairs
    print(f"\nGenerating same-person pairs (max {args.max_same} per person)...")
    same_pairs = generate_same_person_pairs(person_images, args.max_same)
    
    # Generate different-person pairs
    target_diff = args.max_diff if args.max_diff is not None else len(same_pairs)
    print(f"\nGenerating different-person pairs (target: {target_diff})...")
    diff_pairs = generate_different_person_pairs(person_images, target_diff)
    
    # Combine pairs
    all_pairs = same_pairs + diff_pairs
    
    print(f"\nSummary:")
    print(f"  Same-person pairs: {len(same_pairs)}")
    print(f"  Different-person pairs: {len(diff_pairs)}")
    print(f"  Total pairs: {len(all_pairs)}")
    
    # Write CSV
    print(f"\nWriting to: {output_path}")
    write_validation_csv(all_pairs, output_path)
    
    print(f"\n[OK] Validation pairs CSV created successfully!")
    print(f"     Run eval_threshold_sweep.py to evaluate thresholds.")
    
    return 0


if __name__ == "__main__":
    exit(main())

