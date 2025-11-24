"""
Child-Only Threshold Sweep Evaluation Script for Face Recognition Pipeline

This script evaluates thresholds specifically for CHILDREN, as children's faces
are less distinctive and require different thresholds than adults.

HOW TO USE:
-----------
1. Prepare child validation data:
   Create a CSV file at: datasets/validation_pairs_children.csv
   
   Format (CSV with header):
   person_id_1,image_path_1,age_1,person_id_2,image_path_2,age_2,label
   
   Where:
   - person_id_1, person_id_2: Child identifiers
   - image_path_1, image_path_2: Relative paths from repo root
   - age_1, age_2: Age or age group (optional, can be empty)
   - label: 1 for same child, 0 for different children
   
   You can generate this CSV using:
   python tools/build_validation_pairs_children.py

2. Run the script:
   python -m BE.tests.eval_threshold_sweep_children
   
   Or from the BE directory:
   python tests/eval_threshold_sweep_children.py

3. Interpret results:
   - TPR (True Positive Rate): Recall on same-child pairs. Higher is better.
   - FPR (False Positive Rate): Fraction of different-children pairs wrongly accepted. Lower is better.
   - EER (Equal Error Rate): Threshold where FPR ≈ 1 - TPR. Lower EER is better.
   - For missing-person search (high recall needed), choose threshold that balances TPR and FPR.
   - The script will recommend a threshold based on FPR constraint and F1 maximization.

OUTPUT:
-------
- Console table showing TPR/FPR/Precision/F1 for each threshold
- Recommended child threshold for search
- CSV file: tests/threshold_sweep_children_results.csv with detailed metrics
"""

import sys
import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from models.face_detection import FaceDetector
from models.face_embedding import create_embedding_backend
from utils.image_processing import load_image_from_bytes, normalize_image_orientation
from api.config import Settings
from loguru import logger

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# CONFIGURATION (can be overridden via CLI)
# ============================================================================

# Default threshold sweep parameters
MIN_THRESHOLD = 0.30
MAX_THRESHOLD = 0.80
THRESHOLD_STEP = 0.01  # Fine resolution for smooth curves

# Target FPR for recommended threshold (for missing-person search with children)
TARGET_CHILD_FPR = 0.01  # 1% false positive rate (stricter for children)

# Default validation CSV path
DEFAULT_VALIDATION_CSV = "datasets/validation_pairs_children.csv"


# ============================================================================
# HELPER FUNCTIONS (reused from eval_threshold_sweep.py)
# ============================================================================

def extract_embedding(
    image_path: str,
    detector: FaceDetector,
    embedder,
    settings: Settings
) -> Optional[np.ndarray]:
    """
    Extract embedding from a single image using the full pipeline.
    
    Steps:
    1. Read image bytes
    2. Normalize EXIF orientation
    3. Decode to BGR image
    4. Detect faces with confidence threshold
    5. Align face using landmarks
    6. Extract embedding
    
    Returns:
        Embedding vector (512-D) or None if failed
    """
    try:
        repo_root = Path(__file__).parent.parent
        full_path = repo_root / image_path
        
        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            return None
        
        # Read image bytes
        with open(full_path, 'rb') as f:
            image_bytes = f.read()
        
        # Normalize orientation
        normalized_bytes = normalize_image_orientation(image_bytes)
        
        # Decode to BGR
        import cv2
        nparr = np.frombuffer(normalized_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"Failed to decode image: {image_path}")
            return None
        
        # Detect faces
        faces = detector.detect_faces(image, confidence_threshold=settings.face_confidence_threshold)
        
        if not faces:
            logger.warning(f"No face detected in: {image_path}")
            return None
        
        # Use the largest face
        main_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
        
        # Align face
        aligned = detector.align_face(image, main_face['keypoints'])
        
        if aligned is None:
            logger.warning(f"Face alignment failed: {image_path}")
            return None
        
        # Extract embedding
        embedding = embedder.extract_embedding(aligned)
        
        if embedding is None:
            logger.warning(f"Embedding extraction failed: {image_path}")
            return None
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def load_validation_pairs(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load validation pairs from CSV file.
    
    Args:
        csv_path: Path to validation_pairs_children.csv
        
    Returns:
        List of pair dicts with keys: person_id_1, image_path_1, age_1, person_id_2, image_path_2, age_2, label
    """
    pairs = []
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Handle both old format (without age) and new format (with age)
            pair = {
                'person_id_1': row.get('person_id_1', ''),
                'image_path_1': row.get('image_path_1', ''),
                'age_1': row.get('age_1', ''),
                'person_id_2': row.get('person_id_2', ''),
                'image_path_2': row.get('image_path_2', ''),
                'age_2': row.get('age_2', ''),
                'label': int(row.get('label', 0))
            }
            pairs.append(pair)
    
    same_count = sum(1 for p in pairs if p['label'] == 1)
    diff_count = sum(1 for p in pairs if p['label'] == 0)
    
    logger.info(f"Loaded {len(pairs)} validation pairs:")
    logger.info(f"  Same-child pairs: {same_count}")
    logger.info(f"  Different-children pairs: {diff_count}")
    
    return pairs


def compute_embeddings_cache(
    pairs: List[Dict[str, str]],
    detector: FaceDetector,
    embedder,
    settings: Settings
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute and cache embeddings for all unique image paths.
    
    Returns:
        Dict mapping image_path -> embedding (or None if failed)
    """
    # Collect unique image paths
    unique_paths = set()
    for pair in pairs:
        unique_paths.add(pair['image_path_1'])
        unique_paths.add(pair['image_path_2'])
    
    logger.info(f"Computing embeddings for {len(unique_paths)} unique images...")
    
    cache = {}
    failed = 0
    
    for i, image_path in enumerate(sorted(unique_paths)):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(unique_paths)}")
        
        embedding = extract_embedding(image_path, detector, embedder, settings)
        cache[image_path] = embedding
        
        if embedding is None:
            failed += 1
    
    logger.info(f"Embedding computation complete: {len(cache) - failed} succeeded, {failed} failed")
    
    return cache


def compute_pair_similarities(
    pairs: List[Dict[str, str]],
    embedding_cache: Dict[str, Optional[np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities for all pairs.
    
    Returns:
        Tuple of (similarities, labels) as numpy arrays
    """
    similarities = []
    labels = []
    skipped = 0
    
    for pair in pairs:
        emb1 = embedding_cache.get(pair['image_path_1'])
        emb2 = embedding_cache.get(pair['image_path_2'])
        
        if emb1 is None or emb2 is None:
            logger.warning(f"Skipping pair due to missing embeddings: {pair['image_path_1']} vs {pair['image_path_2']}")
            skipped += 1
            continue
        
        # Compute cosine similarity (dot product for L2-normalized embeddings)
        similarity = np.dot(emb1, emb2)
        similarities.append(similarity)
        labels.append(pair['label'])
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} pairs due to missing embeddings")
    
    logger.info(f"Computed similarities for {len(similarities)} pairs")
    
    return np.array(similarities), np.array(labels)


def evaluate_thresholds(
    similarities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float = TARGET_CHILD_FPR
) -> Tuple[List[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Evaluate thresholds and compute metrics.
    
    Returns:
        Tuple of:
        - results: List of dicts with metrics for each threshold
        - eer_info: Dict with EER threshold and value
        - max_f1_info: Dict with threshold that maximizes F1
        - low_fpr_info: Dict with threshold that gives FPR <= target_fpr
        - recommended_info: Dict with recommended threshold for search
    """
    results = []
    
    for threshold in thresholds:
        # Predictions: 1 if similarity >= threshold, else 0
        predictions = (similarities >= threshold).astype(int)
        
        # Confusion matrix
        TP = np.sum((labels == 1) & (predictions == 1))
        FN = np.sum((labels == 1) & (predictions == 0))
        FP = np.sum((labels == 0) & (predictions == 1))
        TN = np.sum((labels == 0) & (predictions == 0))
        
        # Metrics
        TPR = TP / (TP + FN + 1e-8)  # Recall (same-child)
        FPR = FP / (FP + TN + 1e-8)  # False positive rate (different-children)
        FNR = 1 - TPR  # False negative rate
        Precision = TP / (TP + FP + 1e-8)
        F1 = 2 * Precision * TPR / (Precision + TPR + 1e-8) if (Precision + TPR) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'TPR': TPR,
            'FPR': FPR,
            'FNR': FNR,
            'Precision': Precision,
            'F1': F1,
            'TP': TP,
            'FN': FN,
            'FP': FP,
            'TN': TN
        })
    
    # Find EER (where FPR ≈ FNR)
    eer_info = None
    min_diff = float('inf')
    for r in results:
        diff = abs(r['FPR'] - r['FNR'])
        if diff < min_diff:
            min_diff = diff
            eer_info = {
                'threshold': r['threshold'],
                'eer': (r['FPR'] + r['FNR']) / 2,
                'TPR': r['TPR'],
                'FPR': r['FPR'],
                'F1': r['F1']
            }
    
    # Find threshold with max F1
    max_f1_info = None
    max_f1 = -1
    for r in results:
        if r['F1'] > max_f1:
            max_f1 = r['F1']
            max_f1_info = {
                'threshold': r['threshold'],
                'F1': r['F1'],
                'TPR': r['TPR'],
                'FPR': r['FPR'],
                'Precision': r['Precision']
            }
    
    # Find threshold with FPR <= target_fpr
    low_fpr_info = None
    best_f1_at_low_fpr = -1
    for r in results:
        if r['FPR'] <= target_fpr:
            if r['F1'] > best_f1_at_low_fpr:
                best_f1_at_low_fpr = r['F1']
                low_fpr_info = {
                    'threshold': r['threshold'],
                    'FPR': r['FPR'],
                    'F1': r['F1'],
                    'TPR': r['TPR'],
                    'Precision': r['Precision']
                }
    
    # Compute recommended threshold
    # Strategy: Among thresholds with FPR <= target_fpr, pick max F1
    # If no such threshold, fall back to threshold with minimal (FPR + FNR) or max F1
    recommended_info = None
    
    if low_fpr_info:
        # Use threshold with FPR <= target and max F1
        recommended_info = low_fpr_info.copy()
        recommended_info['reason'] = f'FPR <= {target_fpr} with max F1'
    else:
        # Fallback: try with relaxed FPR (2x target)
        relaxed_fpr = target_fpr * 2
        best_f1_relaxed = -1
        for r in results:
            if r['FPR'] <= relaxed_fpr:
                if r['F1'] > best_f1_relaxed:
                    best_f1_relaxed = r['F1']
                    recommended_info = {
                        'threshold': r['threshold'],
                        'FPR': r['FPR'],
                        'F1': r['F1'],
                        'TPR': r['TPR'],
                        'Precision': r['Precision'],
                        'reason': f'FPR <= {relaxed_fpr} (relaxed) with max F1'
                    }
        
        # Final fallback: use max F1 overall
        if recommended_info is None:
            recommended_info = max_f1_info.copy()
            recommended_info['reason'] = 'Max F1 overall (no FPR constraint satisfied)'
    
    return results, eer_info, max_f1_info, low_fpr_info, recommended_info


def print_results_table(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print("CHILD-SPECIFIC THRESHOLD EVALUATION RESULTS")
    print("=" * 120)
    print(f"{'Threshold':<10} {'TPR':<8} {'FPR':<8} {'FNR':<8} {'Precision':<10} {'F1':<8} {'TP':<6} {'FN':<6} {'FP':<6} {'TN':<6}")
    print("-" * 120)
    
    for r in results:
        print(f"{r['threshold']:<10.2f} {r['TPR']:<8.3f} {r['FPR']:<8.3f} {r['FNR']:<8.3f} "
              f"{r['Precision']:<10.3f} {r['F1']:<8.3f} {r['TP']:<6} {r['FN']:<6} {r['FP']:<6} {r['TN']:<6}")
    
    print("=" * 120)


def save_results_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['threshold', 'TPR', 'FPR', 'FNR', 'Precision', 'F1', 'TP', 'FN', 'FP', 'TN']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Saved results to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Child-only threshold sweep evaluation"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=DEFAULT_VALIDATION_CSV,
        help=f"Path to validation_pairs_children.csv (default: {DEFAULT_VALIDATION_CSV})"
    )
    parser.add_argument(
        '--min-threshold',
        type=float,
        default=MIN_THRESHOLD,
        help=f"Minimum threshold (default: {MIN_THRESHOLD})"
    )
    parser.add_argument(
        '--max-threshold',
        type=float,
        default=MAX_THRESHOLD,
        help=f"Maximum threshold (default: {MAX_THRESHOLD})"
    )
    parser.add_argument(
        '--step',
        type=float,
        default=THRESHOLD_STEP,
        help=f"Threshold step (default: {THRESHOLD_STEP})"
    )
    parser.add_argument(
        '--target-fpr',
        type=float,
        default=TARGET_CHILD_FPR,
        help=f"Target FPR for recommended threshold (default: {TARGET_CHILD_FPR})"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV path (default: tests/threshold_sweep_children_results.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 120)
    print("CHILD-ONLY THRESHOLD SWEEP EVALUATION")
    print("=" * 120)
    
    # Get repo root
    repo_root = Path(__file__).parent.parent
    
    # Load validation pairs
    csv_path = repo_root / args.csv
    print(f"\nLoading validation pairs from: {csv_path}")
    pairs = load_validation_pairs(csv_path)
    
    if not pairs:
        print("[ERROR] No validation pairs loaded!")
        return 1
    
    # Initialize pipeline
    print("\nInitializing face recognition pipeline...")
    settings = Settings()
    detector = FaceDetector()
    embedder = create_embedding_backend(
        backend_type="insightface",
        model_name=settings.insightface_model_name,
        use_gpu=settings.use_gpu
    )
    print("[OK] Pipeline initialized")
    
    # Compute embeddings
    print("\nComputing embeddings for all images...")
    embedding_cache = compute_embeddings_cache(pairs, detector, embedder, settings)
    
    # Compute similarities
    print("\nComputing similarities for all pairs...")
    similarities, labels = compute_pair_similarities(pairs, embedding_cache)
    
    if len(similarities) == 0:
        print("[ERROR] No valid similarities computed!")
        return 1
    
    # Generate thresholds
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    print(f"\nEvaluating {len(thresholds)} thresholds from {args.min_threshold} to {args.max_threshold} (step {args.step})...")
    
    # Evaluate thresholds
    results, eer_info, max_f1_info, low_fpr_info, recommended_info = evaluate_thresholds(
        similarities, labels, thresholds, target_fpr=args.target_fpr
    )
    
    # Print results
    print_results_table(results)
    
    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    if eer_info:
        print(f"\nEER (Equal Error Rate):")
        print(f"  Threshold: {eer_info['threshold']:.3f}")
        print(f"  EER: {eer_info['eer']:.3f}")
        print(f"  TPR: {eer_info['TPR']:.3f}, FPR: {eer_info['FPR']:.3f}, F1: {eer_info['F1']:.3f}")
    
    if max_f1_info:
        print(f"\nThreshold with Maximum F1-score:")
        print(f"  Threshold: {max_f1_info['threshold']:.3f}")
        print(f"  F1: {max_f1_info['F1']:.3f}")
        print(f"  TPR: {max_f1_info['TPR']:.3f}, FPR: {max_f1_info['FPR']:.3f}, Precision: {max_f1_info['Precision']:.3f}")
    
    if low_fpr_info:
        print(f"\nThreshold with FPR <= {args.target_fpr}:")
        print(f"  Threshold: {low_fpr_info['threshold']:.3f}")
        print(f"  FPR: {low_fpr_info['FPR']:.3f}")
        print(f"  TPR: {low_fpr_info['TPR']:.3f}, F1: {low_fpr_info['F1']:.3f}, Precision: {low_fpr_info['Precision']:.3f}")
    
    # Print recommended threshold
    print("\n" + "=" * 120)
    print("RECOMMENDED CHILD THRESHOLD FOR MISSING-PERSON SEARCH")
    print("=" * 120)
    if recommended_info:
        print(f"\nThreshold: {recommended_info['threshold']:.3f}")
        print(f"Reason: {recommended_info.get('reason', 'N/A')}")
        print(f"\nMetrics at this threshold:")
        print(f"  TPR (same-child recall): {recommended_info['TPR']:.3f}")
        print(f"  FPR (different-children false accept): {recommended_info['FPR']:.3f}")
        print(f"  Precision: {recommended_info['Precision']:.3f}")
        print(f"  F1-score: {recommended_info['F1']:.3f}")
        print(f"\n>>> Use this threshold value for face_search_threshold_child in Settings <<<")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / "tests" / "threshold_sweep_children_results.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, output_path)
    
    print(f"\n[OK] Evaluation complete! Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

