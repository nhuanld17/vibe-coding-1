"""
Threshold Sweep Evaluation Script for Face Recognition Pipeline

HOW TO USE:
-----------
1. Prepare validation data:
   Create a CSV file at: datasets/validation_pairs.csv
   
   Format (CSV with header):
   person_id_1,image_path_1,person_id_2,image_path_2,label
   
   Where:
   - person_id_1, person_id_2: Person identifiers (can be same or different)
   - image_path_1, image_path_2: Relative paths from repo root (e.g., "datasets/real_val/person_001/img1.jpg")
   - label: 1 for same person, 0 for different person
   
   Example:
   person_id_1,image_path_1,person_id_2,image_path_2,label
   person_001,datasets/FGNET_organized/person_001/age_02.jpg,person_001,datasets/FGNET_organized/person_001/age_33.jpg,1
   person_001,datasets/FGNET_organized/person_001/age_02.jpg,person_002,datasets/FGNET_organized/person_002/age_03.jpg,0

2. Run the script:
   python -m BE.tests.eval_threshold_sweep
   
   Or from the BE directory:
   python tests/eval_threshold_sweep.py

3. Interpret results:
   - TPR (True Positive Rate): Recall on same-person pairs. Higher is better.
   - FPR (False Positive Rate): Fraction of different-person pairs wrongly accepted. Lower is better.
   - EER (Equal Error Rate): Threshold where FPR ≈ 1 - TPR. Lower EER is better.
   - For missing-person search (high recall needed), choose threshold slightly LOWER than EER threshold.
   - For security applications (low false positives), choose threshold slightly HIGHER than EER threshold.

OUTPUT:
-------
- Console table showing TPR/FPR for each threshold
- CSV file: tests/threshold_sweep_results.csv with detailed metrics
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
THRESHOLD_STEP = 0.01  # Finer resolution for smoother curves

# Target FPR for recommended threshold (for missing-person search)
TARGET_FPR_FOR_SEARCH = 0.01  # 1% false positive rate


# ============================================================================
# HELPER FUNCTIONS
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
        512-D normalized embedding or None if extraction fails
    """
    try:
        # Read image
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Normalize orientation
        image_bytes = normalize_image_orientation(image_bytes)
        
        # Decode to BGR
        image = load_image_from_bytes(image_bytes)
        if image is None:
            logger.warning(f"Could not decode image: {image_path}")
            return None
        
        # Detect faces
        faces = detector.detect_faces(
            image,
            confidence_threshold=settings.face_confidence_threshold
        )
        
        if not faces:
            logger.warning(f"No faces detected in: {image_path}")
            return None
        
        # Use the first (highest confidence) face
        landmarks = faces[0]['keypoints']
        
        # Align face
        aligned = detector.align_face(image, landmarks, output_size=(112, 112))
        
        # Extract embedding
        embedding = embedder.extract_embedding(aligned)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def load_validation_pairs(csv_path: str) -> List[Dict[str, str]]:
    """
    Load validation pairs from CSV file.
    
    Expected CSV format:
    person_id_1,image_path_1,person_id_2,image_path_2,label
    
    Where label is 1 (same person) or 0 (different person).
    
    Returns:
        List of dicts with keys: img1, img2, label
    """
    pairs = []
    same_count = 0
    different_count = 0
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Validation pairs CSV not found: {csv_path}\n"
            f"Please create it with format:\n"
            f"person_id_1,image_path_1,person_id_2,image_path_2,label\n"
            f"Where label=1 for same person, label=0 for different person"
        )
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Validate required columns
            required_cols = ['person_id_1', 'image_path_1', 'person_id_2', 'image_path_2', 'label']
            if not all(col in row for col in required_cols):
                logger.warning(f"Skipping row with missing columns: {row}")
                continue
            
            # Parse label
            try:
                label = int(row['label'])
                if label not in [0, 1]:
                    logger.warning(f"Invalid label {label}, must be 0 or 1. Skipping row.")
                    continue
            except ValueError:
                logger.warning(f"Could not parse label '{row['label']}'. Skipping row.")
                continue
            
            # Get image paths (relative to repo root)
            img1 = row['image_path_1'].strip()
            img2 = row['image_path_2'].strip()
            
            # Resolve relative paths
            repo_root = Path(__file__).parent.parent
            img1_abs = repo_root / img1
            img2_abs = repo_root / img2
            
            if not img1_abs.exists():
                logger.warning(f"Image not found: {img1_abs}")
                continue
            
            if not img2_abs.exists():
                logger.warning(f"Image not found: {img2_abs}")
                continue
            
            pairs.append({
                'img1': str(img1_abs),
                'img2': str(img2_abs),
                'label': label
            })
            
            if label == 1:
                same_count += 1
            else:
                different_count += 1
    
    logger.info(f"Loaded {len(pairs)} validation pairs:")
    logger.info(f"  - Same-person pairs: {same_count}")
    logger.info(f"  - Different-person pairs: {different_count}")
    
    return pairs


def compute_embeddings_cache(
    pairs: List[Dict[str, str]],
    detector: FaceDetector,
    embedder,
    settings: Settings
) -> Dict[str, Optional[np.ndarray]]:
    """
    Compute embeddings for all unique images in pairs and cache them.
    
    Returns:
        Dict mapping image_path -> embedding (or None if failed)
    """
    # Collect all unique image paths
    unique_images = set()
    for pair in pairs:
        unique_images.add(pair['img1'])
        unique_images.add(pair['img2'])
    
    logger.info(f"Computing embeddings for {len(unique_images)} unique images...")
    
    cache = {}
    success_count = 0
    fail_count = 0
    
    for img_path in unique_images:
        embedding = extract_embedding(img_path, detector, embedder, settings)
        cache[img_path] = embedding
        
        if embedding is not None:
            success_count += 1
        else:
            fail_count += 1
        
        if (success_count + fail_count) % 10 == 0:
            logger.info(f"  Processed {success_count + fail_count}/{len(unique_images)} images...")
    
    logger.info(f"Embedding extraction complete: {success_count} success, {fail_count} failed")
    
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
        emb1 = embedding_cache.get(pair['img1'])
        emb2 = embedding_cache.get(pair['img2'])
        
        if emb1 is None or emb2 is None:
            logger.warning(f"Skipping pair due to missing embeddings: {pair['img1']} vs {pair['img2']}")
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
    target_fpr: float = TARGET_FPR_FOR_SEARCH
) -> Tuple[List[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Evaluate TPR/FPR/Precision/F1 for each threshold.
    
    Args:
        similarities: Array of cosine similarity scores (N,)
        labels: Array of labels, 1=same person, 0=different person (N,)
        thresholds: Array of thresholds to test
        target_fpr: Target FPR for recommended threshold
    
    Returns:
        Tuple of (results_list, eer_info, f1_max_info, low_fpr_info, recommended_info)
        - results_list: List of dicts with metrics for each threshold
        - eer_info: Dict with 'threshold' and 'eer' or None
        - f1_max_info: Dict with threshold that maximizes F1 or None
        - low_fpr_info: Dict with threshold at target FPR or None
        - recommended_info: Dict with recommended threshold for missing-person search or None
    """
    results = []
    
    for threshold in thresholds:
        # Predict same-person if similarity >= threshold
        predictions = (similarities >= threshold).astype(int)
        
        # Compute confusion matrix
        TP = np.sum((labels == 1) & (predictions == 1))
        FN = np.sum((labels == 1) & (predictions == 0))
        FP = np.sum((labels == 0) & (predictions == 1))
        TN = np.sum((labels == 0) & (predictions == 0))
        
        # Compute rates
        TPR = TP / (TP + FN + 1e-8)  # True Positive Rate (Recall)
        FPR = FP / (FP + TN + 1e-8)  # False Positive Rate
        FNR = FN / (TP + FN + 1e-8)  # False Negative Rate
        Precision = TP / (TP + FP + 1e-8)  # Precision
        F1 = 2 * Precision * TPR / (Precision + TPR + 1e-8)  # F1-score
        
        results.append({
            'threshold': threshold,
            'TPR': TPR,
            'FPR': FPR,
            'FNR': FNR,
            'Precision': Precision,
            'F1': F1,
            'TP': int(TP),
            'FN': int(FN),
            'FP': int(FP),
            'TN': int(TN)
        })
    
    # Compute EER: threshold where FPR ≈ FNR
    eer_info = None
    min_diff = float('inf')
    eer_threshold = None
    eer_value = None
    
    for result in results:
        diff = abs(result['FPR'] - result['FNR'])
        if diff < min_diff:
            min_diff = diff
            eer_threshold = result['threshold']
            eer_value = (result['FPR'] + result['FNR']) / 2.0
    
    if eer_threshold is not None:
        eer_info = {
            'threshold': eer_threshold,
            'eer': eer_value
        }
    
    # Find threshold that maximizes F1
    f1_max_info = None
    max_f1 = -1
    for result in results:
        if result['F1'] > max_f1:
            max_f1 = result['F1']
            f1_max_info = {
                'threshold': result['threshold'],
                'F1': result['F1'],
                'TPR': result['TPR'],
                'FPR': result['FPR'],
                'Precision': result['Precision']
            }
    
    # Find threshold with FPR <= target_fpr (highest TPR among those)
    low_fpr_info = None
    best_tpr_at_target = -1
    for result in results:
        if result['FPR'] <= target_fpr and result['TPR'] > best_tpr_at_target:
            best_tpr_at_target = result['TPR']
            low_fpr_info = {
                'threshold': result['threshold'],
                'FPR': result['FPR'],
                'TPR': result['TPR'],
                'Precision': result['Precision'],
                'F1': result['F1']
            }
    
    # Recommended threshold for missing-person search:
    # Among thresholds with FPR <= target_fpr, pick the one with max TPR (or max F1)
    # If no threshold meets target_fpr, use the one with lowest FPR that still has reasonable TPR
    recommended_info = None
    
    if low_fpr_info:
        # Use the threshold that meets target FPR
        recommended_info = low_fpr_info.copy()
        recommended_info['reason'] = f'FPR <= {target_fpr}'
    else:
        # Fallback: find threshold with FPR <= 0.02 (2%) and max TPR
        for result in sorted(results, key=lambda x: x['threshold'], reverse=True):
            if result['FPR'] <= 0.02 and result['TPR'] >= 0.5:
                recommended_info = {
                    'threshold': result['threshold'],
                    'FPR': result['FPR'],
                    'TPR': result['TPR'],
                    'Precision': result['Precision'],
                    'F1': result['F1'],
                    'reason': 'FPR <= 0.02 (fallback)'
                }
                break
        
        # If still no good threshold, use F1-max
        if not recommended_info and f1_max_info:
            recommended_info = f1_max_info.copy()
            recommended_info['reason'] = 'F1-max (fallback)'
    
    return results, eer_info, f1_max_info, low_fpr_info, recommended_info


def print_results_table(
    results: List[Dict],
    eer_info: Optional[Dict],
    f1_max_info: Optional[Dict],
    low_fpr_info: Optional[Dict],
    recommended_info: Optional[Dict]
):
    """Print formatted results table to console."""
    print("\n" + "=" * 120)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 120)
    print(f"{'Threshold':<12} | {'TPR':<8} | {'FPR':<8} | {'Precision':<10} | {'F1':<8} | {'TP':<6} | {'FN':<6} | {'FP':<6} | {'TN':<6}")
    print("-" * 120)
    
    for r in results:
        print(
            f"{r['threshold']:<12.2f} | "
            f"{r['TPR']:<8.4f} | "
            f"{r['FPR']:<8.4f} | "
            f"{r['Precision']:<10.4f} | "
            f"{r['F1']:<8.4f} | "
            f"{r['TP']:<6} | "
            f"{r['FN']:<6} | "
            f"{r['FP']:<6} | "
            f"{r['TN']:<6}"
        )
    
    print("=" * 120)
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    
    if eer_info:
        print(f"\nEER (Equal Error Rate): {eer_info['eer']:.4f} at threshold = {eer_info['threshold']:.3f}")
    
    if f1_max_info:
        print(f"\nF1-Max Threshold: {f1_max_info['threshold']:.3f}")
        print(f"  F1-score: {f1_max_info['F1']:.4f}")
        print(f"  TPR: {f1_max_info['TPR']:.4f}, FPR: {f1_max_info['FPR']:.4f}, Precision: {f1_max_info['Precision']:.4f}")
    
    if low_fpr_info:
        print(f"\nLow FPR Threshold (FPR <= {TARGET_FPR_FOR_SEARCH}): {low_fpr_info['threshold']:.3f}")
        print(f"  FPR: {low_fpr_info['FPR']:.4f}, TPR: {low_fpr_info['TPR']:.4f}")
        print(f"  Precision: {low_fpr_info['Precision']:.4f}, F1: {low_fpr_info['F1']:.4f}")
    else:
        print(f"\nNo threshold found with FPR <= {TARGET_FPR_FOR_SEARCH}")
    
    if recommended_info:
        print(f"\n{'='*120}")
        print(f"RECOMMENDED SEARCH THRESHOLD (Missing-Person Use Case): {recommended_info['threshold']:.3f}")
        print(f"{'='*120}")
        print(f"  Reason: {recommended_info.get('reason', 'N/A')}")
        print(f"  TPR (Recall): {recommended_info['TPR']:.4f}")
        print(f"  FPR: {recommended_info['FPR']:.4f}")
        print(f"  Precision: {recommended_info['Precision']:.4f}")
        print(f"  F1-score: {recommended_info['F1']:.4f}")
        print(f"\n  Use this value for: api.config.Settings.face_search_threshold")
        print(f"  Or set via environment variable: FACE_SEARCH_THRESHOLD={recommended_info['threshold']:.3f}")
    
    print("=" * 120)


def save_results_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['threshold', 'TPR', 'FPR', 'FNR', 'Precision', 'F1', 'TP', 'FN', 'FP', 'TN']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    logger.info(f"Results saved to: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition thresholds with TPR/FPR/Precision/F1 metrics"
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
        default=TARGET_FPR_FOR_SEARCH,
        help=f"Target FPR for recommended threshold (default: {TARGET_FPR_FOR_SEARCH})"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV path (default: tests/threshold_sweep_results.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 120)
    print("FACE RECOGNITION THRESHOLD SWEEP EVALUATION")
    print("=" * 120)
    
    # Configuration
    repo_root = Path(__file__).parent.parent
    validation_csv = repo_root / "datasets" / "validation_pairs.csv"
    
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = Path(__file__).parent / "threshold_sweep_results.csv"
    
    # Threshold range
    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step, args.step)
    
    # Initialize models
    print("\nInitializing models...")
    settings = Settings()
    detector = FaceDetector(min_face_size=40, device="GPU:0" if settings.use_gpu else "CPU:0")
    embedder = create_embedding_backend(
        backend_type=settings.embedding_backend,
        use_gpu=settings.use_gpu,
        model_name=settings.insightface_model_name
    )
    print("[OK] Models initialized")
    
    # Load validation pairs
    print(f"\nLoading validation pairs from: {validation_csv}")
    try:
        pairs = load_validation_pairs(str(validation_csv))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    
    if len(pairs) == 0:
        print("[ERROR] No valid pairs loaded. Please check your CSV file.")
        return 1
    
    # Compute embeddings (with caching)
    print("\nComputing embeddings...")
    embedding_cache = compute_embeddings_cache(pairs, detector, embedder, settings)
    
    # Compute similarities
    print("\nComputing pair similarities...")
    similarities, labels = compute_pair_similarities(pairs, embedding_cache)
    
    if len(similarities) == 0:
        print("[ERROR] No valid similarities computed. Check your data.")
        return 1
    
    # Evaluate thresholds
    print(f"\nEvaluating {len(thresholds)} thresholds from {thresholds[0]:.3f} to {thresholds[-1]:.3f} (step {args.step})...")
    results, eer_info, f1_max_info, low_fpr_info, recommended_info = evaluate_thresholds(
        similarities, labels, thresholds, target_fpr=args.target_fpr
    )
    
    # Print results
    print_results_table(results, eer_info, f1_max_info, low_fpr_info, recommended_info)
    
    # Save CSV
    save_results_csv(results, str(output_csv))
    
    print(f"\n[OK] Evaluation complete!")
    print(f"     Results saved to: {output_csv}")
    
    return 0


if __name__ == "__main__":
    exit(main())

