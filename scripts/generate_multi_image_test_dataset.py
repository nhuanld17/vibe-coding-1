"""
Generate comprehensive multi-image test dataset from FGNetOrganized.

This script creates 250 test pairs with:
- Mix of same/different persons (60% same, 40% different)
- Variable image counts (1-10 images per side)
- Age gap diversity
- Some images without detectable faces

Author: AI Face Recognition Team
"""

import os
import random
from pathlib import Path
import csv
from typing import List, Dict, Tuple, Optional
from loguru import logger

# FGNetOrganized structure:
# FGNetOrganized/
#   001A/ (person 001, age range A)
#     1a001.jpg, 1a002.jpg, ...
#   001B/ (person 001, age range B - older)
#     1a010.jpg, 1a011.jpg, ...
#   002A/
#     2a001.jpg, ...

# Use relative path from BE directory
FGNET_PATH = Path(__file__).parent.parent / "datasets" / "FGNET_organized"
OUTPUT_CSV = Path(__file__).parent.parent / "tests" / "data" / "multi_image_test_dataset.csv"


def get_person_folders() -> Dict[str, Path]:
    """
    Get all person folders.
    
    Dataset structure: person_001/, person_002/, ...
    
    Returns:
        {"001": person_001_path, "002": person_002_path, ...}
    """
    person_folders = {}
    
    if not FGNET_PATH.exists():
        logger.error(f"FGNetOrganized path not found: {FGNET_PATH}")
        logger.info("Please update FGNET_PATH in the script to point to your FGNetOrganized dataset")
        return {}
    
    for folder in FGNET_PATH.iterdir():
        if folder.is_dir() and folder.name.startswith("person_"):
            person_id = folder.name.replace("person_", "").lstrip("0")  # "001" -> "1"
            if person_id == "":
                person_id = "0"
            person_folders[person_id] = folder
    
    return person_folders


def get_images_from_folder(folder: Path, count: Optional[int] = None) -> List[str]:
    """Get random images from folder."""
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
    
    if not images:
        return []
    
    if count is None or count >= len(images):
        return [str(img.relative_to(FGNET_PATH)) for img in images]
    
    selected = random.sample(images, count)
    return [str(img.relative_to(FGNET_PATH)) for img in selected]


def extract_age_from_filename(filename: str) -> int:
    """Extract age from filename like 'age_02.jpg' -> 2"""
    try:
        # Extract number after "age_"
        age_str = filename.replace("age_", "").replace(".jpg", "").replace(".JPG", "")
        return int(age_str)
    except:
        return 0

def calculate_age_gap_from_images(images1: List[str], images2: List[str]) -> Tuple[int, str]:
    """
    Calculate age gap between two image sets.
    
    Uses average age from filenames.
    """
    ages1 = [extract_age_from_filename(img) for img in images1]
    ages2 = [extract_age_from_filename(img) for img in images2]
    
    if not ages1 or not ages2:
        return 0, "N/A"
    
    avg_age1 = sum(ages1) / len(ages1)
    avg_age2 = sum(ages2) / len(ages2)
    gap = abs(avg_age1 - avg_age2)
    
    if gap <= 10:
        category = "small"
    elif gap <= 30:
        category = "medium"
    elif gap <= 50:
        category = "large"
    else:
        category = "xlarge"
    
    return int(gap), category


def generate_same_person_pair(
    person_id: str,
    folder: Path,
    pair_id: int,
    target_category: Optional[str] = None
) -> Optional[Dict]:
    """
    Generate a same-person test pair with age gap.
    
    Args:
        target_category: If specified, try to create pair with this age gap category
                        ("small", "medium", "large", "xlarge")
    """
    
    # Get all images from person folder
    all_images = list(folder.glob("age_*.jpg")) + list(folder.glob("age_*.JPG"))
    
    if len(all_images) < 2:
        return None
    
    # Sort by age
    all_images.sort(key=lambda x: extract_age_from_filename(x.name))
    
    # Extract ages
    ages = [extract_age_from_filename(img.name) for img in all_images]
    min_age = min(ages)
    max_age = max(ages)
    age_range = max_age - min_age
    
    # Strategy based on target category
    if target_category == "large" or target_category == "xlarge":
        # For large gaps: use youngest vs oldest images
        # Need at least 30+ year gap
        if age_range < 30:
            return None  # Skip if person doesn't have large age range
        
        # Use first 30% (youngest) and last 30% (oldest)
        young_count = max(1, len(all_images) // 3)
        old_count = max(1, len(all_images) // 3)
        
        younger_images = all_images[:young_count]
        older_images = all_images[-old_count:]
        
    elif target_category == "medium":
        # For medium gaps: use first 40% vs last 40%
        if age_range < 10:
            return None
        
        young_count = max(1, int(len(all_images) * 0.4))
        old_count = max(1, int(len(all_images) * 0.4))
        
        younger_images = all_images[:young_count]
        older_images = all_images[-old_count:]
        
    else:
        # Default: split in half (small gap)
        mid_point = len(all_images) // 2
        younger_images = all_images[:mid_point] if mid_point > 0 else all_images[:len(all_images)//2]
        older_images = all_images[mid_point:] if mid_point < len(all_images) else all_images[len(all_images)//2:]
    
    if not younger_images or not older_images:
        # Fallback: random split
        random.shuffle(all_images)
        split = len(all_images) // 2
        younger_images = all_images[:split]
        older_images = all_images[split:]
    
    # Variable image counts (1-10)
    query_count = random.randint(1, min(10, len(younger_images)))
    candidate_count = random.randint(1, min(10, len(older_images)))
    
    query_selected = random.sample(younger_images, query_count)
    candidate_selected = random.sample(older_images, candidate_count)
    
    query_images = [str(img.relative_to(FGNET_PATH)) for img in query_selected]
    candidate_images = [str(img.relative_to(FGNET_PATH)) for img in candidate_selected]
    
    gap, category = calculate_age_gap_from_images(
        [img.name for img in query_selected],
        [img.name for img in candidate_selected]
    )
    
    # Verify category matches target (if specified)
    if target_category and category != target_category:
        # If doesn't match, return None to try again
        return None
    
    return {
        "pair_id": pair_id,
        "person_id": person_id,
        "is_same_person": True,
        "query_images": "|".join(query_images),
        "candidate_images": "|".join(candidate_images),
        "age_gap_category": category,
        "age_gap_years": gap,
        "query_image_count": len(query_images),
        "candidate_image_count": len(candidate_images),
        "has_no_face_query": False,
        "has_no_face_candidate": False,
        "notes": f"Same person, {category} age gap (~{gap} years)"
    }


def generate_different_person_pair(
    person_folders: Dict[str, Path],
    pair_id: int
) -> Optional[Dict]:
    """Generate a different-person test pair."""
    
    if len(person_folders) < 2:
        return None
    
    # Select 2 different persons
    person_ids = random.sample(list(person_folders.keys()), 2)
    
    folder1 = person_folders[person_ids[0]]
    folder2 = person_folders[person_ids[1]]
    
    folder1_images = list(folder1.glob("age_*.jpg")) + list(folder1.glob("age_*.JPG"))
    folder2_images = list(folder2.glob("age_*.jpg")) + list(folder2.glob("age_*.JPG"))
    
    if not folder1_images or not folder2_images:
        return None
    
    query_count = random.randint(1, min(10, len(folder1_images)))
    candidate_count = random.randint(1, min(10, len(folder2_images)))
    
    query_selected = random.sample(folder1_images, query_count)
    candidate_selected = random.sample(folder2_images, candidate_count)
    
    query_images = [str(img.relative_to(FGNET_PATH)) for img in query_selected]
    candidate_images = [str(img.relative_to(FGNET_PATH)) for img in candidate_selected]
    
    return {
        "pair_id": pair_id,
        "person_id": f"{person_ids[0]}_vs_{person_ids[1]}",
        "is_same_person": False,
        "query_images": "|".join(query_images),
        "candidate_images": "|".join(candidate_images),
        "age_gap_category": "N/A",
        "age_gap_years": 0,
        "query_image_count": len(query_images),
        "candidate_image_count": len(candidate_images),
        "has_no_face_query": False,
        "has_no_face_candidate": False,
        "notes": f"Different persons ({person_ids[0]} vs {person_ids[1]})"
    }


def generate_dataset():
    """Generate 250 test pairs."""
    
    logger.info("Loading FGNetOrganized structure...")
    person_folders = get_person_folders()
    
    if not person_folders:
        logger.error("No person folders found. Please check FGNET_PATH.")
        return
    
    logger.info(f"Found {len(person_folders)} persons")
    
    # Target: 250 pairs (150 same person, 100 different person)
    TARGET_SAME = 150
    TARGET_DIFFERENT = 100
    
    test_pairs = []
    pair_id = 1
    
    # Generate same-person pairs with target distribution
    logger.info("Generating same-person pairs...")
    # Need persons with at least 2 images for age gap
    persons_with_multiple_images = [
        (pid, folder) for pid, folder in person_folders.items()
        if len(list(folder.glob("age_*.jpg")) + list(folder.glob("age_*.JPG"))) >= 2
    ]
    
    # Find persons with large age ranges (for large gap pairs)
    persons_with_large_range = []
    for pid, folder in persons_with_multiple_images:
        images = list(folder.glob("age_*.jpg")) + list(folder.glob("age_*.JPG"))
        ages = [extract_age_from_filename(img.name) for img in images]
        if len(ages) >= 2:
            age_range = max(ages) - min(ages)
            if age_range >= 30:  # At least 30 year range
                persons_with_large_range.append((pid, folder))
    
    if not persons_with_multiple_images:
        logger.warning("No persons with multiple images found!")
        return
    
    # Target distribution: ~20 large, ~50 medium, ~80 small
    TARGET_LARGE = 20
    TARGET_MEDIUM = 50
    TARGET_SMALL = TARGET_SAME - TARGET_LARGE - TARGET_MEDIUM
    
    logger.info(f"Target distribution: {TARGET_SMALL} small, {TARGET_MEDIUM} medium, {TARGET_LARGE} large")
    
    # Generate large gap pairs first (harder to find)
    logger.info("Generating large age gap pairs...")
    attempts = 0
    max_attempts = TARGET_LARGE * 20
    
    while len([p for p in test_pairs if p and p.get("age_gap_category") in ["large", "xlarge"]]) < TARGET_LARGE and attempts < max_attempts:
        if persons_with_large_range:
            person_id, folder = random.choice(persons_with_large_range)
        else:
            person_id, folder = random.choice(persons_with_multiple_images)
        pair = generate_same_person_pair(person_id, folder, pair_id, target_category="large")
        if pair:
            test_pairs.append(pair)
            pair_id += 1
        attempts += 1
    
    large_count = len([p for p in test_pairs if p and p.get("age_gap_category") in ["large", "xlarge"]])
    logger.info(f"Generated {large_count} large age gap pairs")
    
    # Generate medium gap pairs
    logger.info("Generating medium age gap pairs...")
    attempts = 0
    max_attempts = TARGET_MEDIUM * 10
    
    while len([p for p in test_pairs if p and p.get("age_gap_category") == "medium"]) < TARGET_MEDIUM and attempts < max_attempts:
        person_id, folder = random.choice(persons_with_multiple_images)
        pair = generate_same_person_pair(person_id, folder, pair_id, target_category="medium")
        if pair:
            test_pairs.append(pair)
            pair_id += 1
        attempts += 1
    
    medium_count = len([p for p in test_pairs if p and p.get("age_gap_category") == "medium"])
    logger.info(f"Generated {medium_count} medium age gap pairs")
    
    # Generate remaining as small gap pairs
    logger.info("Generating small age gap pairs...")
    attempts = 0
    max_attempts = TARGET_SMALL * 10
    
    while len([p for p in test_pairs if p and p["is_same_person"]]) < TARGET_SAME and attempts < max_attempts:
        person_id, folder = random.choice(persons_with_multiple_images)
        pair = generate_same_person_pair(person_id, folder, pair_id, target_category="small")
        if pair:
            test_pairs.append(pair)
            pair_id += 1
        attempts += 1
    
    same_count = len([p for p in test_pairs if p and p["is_same_person"]])
    logger.info(f"Generated {same_count} same-person pairs")
    
    # Generate different-person pairs
    logger.info("Generating different-person pairs...")
    attempts = 0
    max_attempts = TARGET_DIFFERENT * 10
    
    while len([p for p in test_pairs if not p["is_same_person"]]) < TARGET_DIFFERENT and attempts < max_attempts:
        pair = generate_different_person_pair(person_folders, pair_id)
        if pair:
            test_pairs.append(pair)
            pair_id += 1
        attempts += 1
    
    diff_count = len([p for p in test_pairs if not p["is_same_person"]])
    logger.info(f"Generated {diff_count} different-person pairs")
    
    # Shuffle
    random.shuffle(test_pairs)
    
    # Save to CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "pair_id", "person_id", "is_same_person",
        "query_images", "candidate_images",
        "age_gap_category", "age_gap_years",
        "query_image_count", "candidate_image_count",
        "has_no_face_query", "has_no_face_candidate",
        "notes"
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_pairs)
    
    logger.success(f"Dataset saved to {OUTPUT_CSV}")
    logger.info(f"Total pairs: {len(test_pairs)}")
    
    # Statistics
    same_count = len([p for p in test_pairs if p["is_same_person"]])
    diff_count = len(test_pairs) - same_count
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total pairs: {len(test_pairs)}")
    print(f"Same person: {same_count} ({same_count/len(test_pairs)*100:.1f}%)")
    print(f"Different person: {diff_count} ({diff_count/len(test_pairs)*100:.1f}%)")
    print("\nAge gap distribution (same-person only):")
    for cat in ["small", "medium", "large", "xlarge"]:
        count = len([p for p in test_pairs if p.get("age_gap_category") == cat])
        if count > 0:
            print(f"  {cat:8s}: {count}")
    print("="*60)


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    generate_dataset()

