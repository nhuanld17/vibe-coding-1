"""
Comprehensive multi-image logic test WITHOUT API server.

Tests PURE LOGIC:
- Face detection
- Embedding extraction
- Multi-image aggregation
- Matching accuracy

NO dependencies on:
- FastAPI server
- Cloudinary
- Qdrant vector DB

Author: AI Face Recognition Team
"""

import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from loguru import logger
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.face_detection import FaceDetector
from models.face_embedding_insightface import InsightFaceEmbedder
from services.multi_image_aggregation import get_aggregation_service
from utils.image_helpers import calculate_age_at_photo

# Paths (relative to BE directory)
FGNET_PATH = Path(__file__).parent.parent / "datasets" / "FGNET_organized"
TEST_CSV = Path(__file__).parent / "data" / "multi_image_test_dataset.csv"
RESULTS_CSV = Path(__file__).parent / "data" / "multi_image_test_results.csv"

# Initialize models (ONCE, reuse for all tests)
logger.info("Initializing models...")
try:
    face_detector = FaceDetector()
    embedding_extractor = InsightFaceEmbedder()
    aggregation_service = get_aggregation_service()
    logger.success("Models ready!")
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    logger.error("Please ensure models are properly configured")
    raise


def process_image(image_path: Path) -> Dict:
    """
    Process single image: detect face, extract embedding.
    
    Returns:
        {
            "success": bool,
            "embedding": np.array or None,
            "validation_status": str,
            "quality_score": float
        }
    """
    try:
        # Read image
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                "success": False,
                "embedding": None,
                "validation_status": "file_read_error",
                "quality_score": 0.0
            }
        
        # Detect face
        face = face_detector.extract_largest_face(img)
        
        if face is None:
            return {
                "success": True,
                "embedding": None,
                "validation_status": "no_face_detected",
                "quality_score": 0.0
            }
        
        # Check quality
        is_good_quality, quality_metrics = face_detector.check_face_quality(face)
        quality_score = quality_metrics.get('quality_score', 0.0)
        
        if quality_score < 0.60:
            return {
                "success": True,
                "embedding": None,
                "validation_status": "low_quality",
                "quality_score": quality_score
            }
        
        # Extract embedding
        embedding = embedding_extractor.extract_embedding(face)
        
        return {
            "success": True,
            "embedding": embedding,
            "validation_status": "valid",
            "quality_score": quality_score
        }
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            "success": False,
            "embedding": None,
            "validation_status": "processing_error",
            "quality_score": 0.0
        }


def process_image_set(image_paths: List[str]) -> Tuple[List[Dict], int, int]:
    """
    Process a set of images.
    
    Returns:
        (valid_images, reference_images_count, failed_images_count)
        valid_images: List of dicts with 'embedding', 'quality', 'image_path'
    """
    valid_images = []
    reference_count = 0
    failed_count = 0
    
    for img_path_str in image_paths:
        full_path = FGNET_PATH / img_path_str
        
        if not full_path.exists():
            logger.warning(f"Image not found: {full_path}")
            failed_count += 1
            continue
        
        result = process_image(full_path)
        
        if not result["success"]:
            failed_count += 1
        elif result["embedding"] is not None:
            valid_images.append({
                "embedding": result["embedding"],
                "quality": result["quality_score"],
                "image_path": img_path_str  # Keep path for age extraction
            })
        else:
            reference_count += 1
    
    return valid_images, reference_count, failed_count


def test_single_pair(pair: Dict) -> Dict:
    """
    Test a single pair from dataset.
    
    Returns result dict with predictions and metrics.
    """
    pair_id = pair["pair_id"]
    is_same_person = pair["is_same_person"] == "True" or pair["is_same_person"] == True
    
    logger.info(f"Testing pair {pair_id}...")
    
    # Parse image paths
    query_paths = pair["query_images"].split("|")
    candidate_paths = pair["candidate_images"].split("|")
    
    start_time = time.time()
    
    # Process query images
    query_valid_raw, query_ref, query_fail = process_image_set(query_paths)
    
    # Process candidate images
    cand_valid_raw, cand_ref, cand_fail = process_image_set(candidate_paths)
    
    processing_time = (time.time() - start_time) * 1000
    
    # Check if we can match
    can_match = len(query_valid_raw) > 0 and len(cand_valid_raw) > 0
    
    if not can_match:
        return {
            **pair,
            "can_match": False,
            "query_valid_count": len(query_valid_raw),
            "query_reference_count": query_ref,
            "query_failed_count": query_fail,
            "candidate_valid_count": len(cand_valid_raw),
            "candidate_reference_count": cand_ref,
            "candidate_failed_count": cand_fail,
            "best_similarity": 0.0,
            "mean_similarity": 0.0,
            "consistency_score": 0.0,
            "predicted_same_person": False,
            "prediction_correct": None,
            "processing_time_ms": processing_time,
            "notes": pair.get("notes", "") + " | Cannot match: no valid embeddings"
        }
    
    # Build query_images and target_images in correct format for aggregation
    # Format needed: {'image_id': str, 'embedding': np.array, 'age_at_photo': int, 'case_id': str}
    query_images = []
    for idx, img_data in enumerate(query_valid_raw):
        # Extract age from image path
        img_path = img_data.get("image_path", "")
        age = 0
        if "age_" in img_path:
            try:
                # Extract from path like "person_066\age_02.jpg" -> 2
                age_str = img_path.split("age_")[1].split(".")[0].split("\\")[0]
                age = int(age_str)
            except:
                age = 0
        
        query_images.append({
            "image_id": f"query_{pair_id}_img_{idx}",
            "embedding": img_data["embedding"],
            "age_at_photo": age,
            "case_id": f"query_{pair_id}"
        })
    
    target_images = []
    for idx, img_data in enumerate(cand_valid_raw):
        # Extract age from image path
        img_path = img_data.get("image_path", "")
        age = 0
        if "age_" in img_path:
            try:
                age_str = img_path.split("age_")[1].split(".")[0].split("\\")[0]
                age = int(age_str)
            except:
                age = 0
        
        target_images.append({
            "image_id": f"target_{pair_id}_img_{idx}",
            "embedding": img_data["embedding"],
            "age_at_photo": age,
            "case_id": f"target_{pair_id}"
        })
    
    # Run aggregation
    try:
        agg_result = aggregation_service.aggregate_multi_image_similarity(
            query_images=query_images,
            target_images=target_images
        )
    except Exception as e:
        logger.error(f"Aggregation failed for pair {pair_id}: {e}")
        return {
            **pair,
            "can_match": False,
            "query_valid_count": len(query_valid_raw),
            "candidate_valid_count": len(cand_valid_raw),
            "best_similarity": 0.0,
            "predicted_same_person": False,
            "prediction_correct": None,
            "processing_time_ms": processing_time,
            "notes": f"Aggregation error: {str(e)}"
        }
    
    # Predict same/different based on threshold
    # agg_result is AggregatedMatchResult object, access as attributes
    THRESHOLD = 0.30  # Adjust based on your system
    predicted_same = agg_result.best_similarity >= THRESHOLD
    prediction_correct = (predicted_same == is_same_person)
    
    return {
        **pair,
        "can_match": True,
        "query_valid_count": len(query_valid_raw),
        "query_reference_count": query_ref,
        "query_failed_count": query_fail,
        "candidate_valid_count": len(cand_valid_raw),
        "candidate_reference_count": cand_ref,
        "candidate_failed_count": cand_fail,
        "best_similarity": agg_result.best_similarity,
        "mean_similarity": agg_result.mean_similarity,
        "consistency_score": agg_result.consistency_score,
        "num_comparisons": len(agg_result.all_pair_scores),
        "predicted_same_person": predicted_same,
        "prediction_correct": prediction_correct,
        "processing_time_ms": processing_time
    }


def run_comprehensive_test():
    """Run test on all 250 pairs."""
    logger.info(f"Loading test dataset from {TEST_CSV}...")
    
    if not TEST_CSV.exists():
        logger.error(f"Test dataset not found: {TEST_CSV}")
        logger.info("Please run scripts/generate_multi_image_test_dataset.py first")
        return
    
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_pairs = list(reader)
    
    logger.info(f"Loaded {len(test_pairs)} test pairs")
    
    # Run tests
    results = []
    start_time = time.time()
    
    for i, pair in enumerate(test_pairs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Progress: {i}/{len(test_pairs)}")
        
        result = test_single_pair(pair)
        results.append(result)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(test_pairs) - i) * avg_time
            logger.info(f"Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min")
    
    total_time = time.time() - start_time
    
    # Save results
    logger.info(f"\nSaving results to {RESULTS_CSV}...")
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        # Original fields
        "pair_id", "person_id", "is_same_person",
        "query_images", "candidate_images",
        "age_gap_category", "age_gap_years",
        "query_image_count", "candidate_image_count",
        "has_no_face_query", "has_no_face_candidate",
        
        # NEW: Test results
        "can_match",
        "query_valid_count", "query_reference_count", "query_failed_count",
        "candidate_valid_count", "candidate_reference_count", "candidate_failed_count",
        "best_similarity", "mean_similarity", "consistency_score", "num_comparisons",
        "predicted_same_person", "prediction_correct",
        "processing_time_ms",
        "notes"
    ]
    
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.success(f"Results saved to {RESULTS_CSV}")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    matchable = [r for r in results if r.get("can_match") == True or r.get("can_match") == "True"]
    correct = [r for r in matchable if r.get("prediction_correct") == True]
    
    print(f"Total pairs tested: {len(results)}")
    print(f"Matchable pairs: {len(matchable)} ({len(matchable)/len(results)*100:.1f}%)")
    print(f"Unmatchable pairs: {len(results) - len(matchable)}")
    
    if len(matchable) > 0:
        print(f"\nOverall Accuracy: {len(correct)}/{len(matchable)} = {len(correct)/len(matchable)*100:.2f}%")
        
        # Breakdown by ground truth
        same_person_pairs = [r for r in matchable if r.get("is_same_person") == "True" or r.get("is_same_person") == True]
        diff_person_pairs = [r for r in matchable if r.get("is_same_person") == "False" or r.get("is_same_person") == False]
        
        same_correct = [r for r in same_person_pairs if r.get("prediction_correct") == True]
        diff_correct = [r for r in diff_person_pairs if r.get("prediction_correct") == True]
        
        print(f"\nSame Person Pairs:")
        print(f"  Total: {len(same_person_pairs)}")
        if len(same_person_pairs) > 0:
            print(f"  Correct: {len(same_correct)} ({len(same_correct)/len(same_person_pairs)*100:.2f}%)")
        
        print(f"\nDifferent Person Pairs:")
        print(f"  Total: {len(diff_person_pairs)}")
        if len(diff_person_pairs) > 0:
            print(f"  Correct: {len(diff_correct)} ({len(diff_correct)/len(diff_person_pairs)*100:.2f}%)")
        
        # Age gap breakdown
        print(f"\nAccuracy by Age Gap (same-person pairs only):")
        for category in ["small", "medium", "large", "xlarge"]:
            cat_pairs = [r for r in same_person_pairs if r.get("age_gap_category") == category]
            if cat_pairs:
                cat_correct = [r for r in cat_pairs if r.get("prediction_correct") == True]
                print(f"  {category:8s}: {len(cat_correct)}/{len(cat_pairs)} = {len(cat_correct)/len(cat_pairs)*100:.2f}%")
    
    # Performance
    avg_processing = sum(float(r.get("processing_time_ms", 0)) for r in results) / len(results) if results else 0
    print(f"\nPerformance:")
    print(f"  Total time: {total_time/60:.2f} minutes")
    print(f"  Avg per pair: {avg_processing:.2f}ms")
    print("="*70)


if __name__ == "__main__":
    run_comprehensive_test()

