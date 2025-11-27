"""
Analyze multi-image test results.

Generates:
- Confusion matrix
- Precision, Recall, F1
- Similarity distribution plots
- Age gap accuracy breakdown

Author: AI Face Recognition Team
"""

import csv
from pathlib import Path
from typing import List, Dict
from loguru import logger

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("pandas/matplotlib not available. Skipping plots.")

RESULTS_CSV = Path("tests/data/multi_image_test_results.csv")


def analyze_results():
    """Analyze test results and generate report."""
    
    if not RESULTS_CSV.exists():
        logger.error(f"Results file not found: {RESULTS_CSV}")
        logger.info("Please run tests/test_multi_image_logic_comprehensive.py first")
        return
    
    logger.info(f"Loading results from {RESULTS_CSV}...")
    
    # Load results
    results = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    logger.info(f"Loaded {len(results)} results")
    
    # Convert to boolean
    for r in results:
        r["is_same_person"] = r["is_same_person"] == "True" or r["is_same_person"] == True
        r["predicted_same_person"] = r["predicted_same_person"] == "True" or r["predicted_same_person"] == True
        r["can_match"] = r.get("can_match") == "True" or r.get("can_match") == True
        r["prediction_correct"] = r.get("prediction_correct") == "True" or r.get("prediction_correct") == True
        try:
            r["best_similarity"] = float(r.get("best_similarity", 0))
        except (ValueError, TypeError):
            r["best_similarity"] = 0.0
    
    # Filter matchable
    matchable = [r for r in results if r["can_match"]]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST RESULTS ANALYSIS")
    print("="*70)
    
    print(f"\nTotal pairs tested: {len(results)}")
    print(f"Matchable pairs: {len(matchable)} ({len(matchable)/len(results)*100:.1f}%)")
    print(f"Unmatchable pairs: {len(results) - len(matchable)}")
    
    if len(matchable) == 0:
        logger.warning("No matchable pairs found!")
        return
    
    # Calculate confusion matrix
    tp = len([r for r in matchable if r["is_same_person"] and r["predicted_same_person"]])
    fn = len([r for r in matchable if r["is_same_person"] and not r["predicted_same_person"]])
    tn = len([r for r in matchable if not r["is_same_person"] and not r["predicted_same_person"]])
    fp = len([r for r in matchable if not r["is_same_person"] and r["predicted_same_person"]])
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(f"True Positives (Correct Same):  {tp:4d}")
    print(f"False Negatives (Missed Same):  {fn:4d}")
    print(f"True Negatives (Correct Diff):   {tn:4d}")
    print(f"False Positives (False Match):   {fp:4d}")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(matchable) if len(matchable) > 0 else 0
    
    print("\n" + "="*70)
    print("METRICS")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Breakdown by ground truth
    same_person_pairs = [r for r in matchable if r["is_same_person"]]
    diff_person_pairs = [r for r in matchable if not r["is_same_person"]]
    
    same_correct = [r for r in same_person_pairs if r["prediction_correct"]]
    diff_correct = [r for r in diff_person_pairs if r["prediction_correct"]]
    
    print("\n" + "="*70)
    print("BREAKDOWN BY GROUND TRUTH")
    print("="*70)
    print(f"\nSame Person Pairs:")
    print(f"  Total: {len(same_person_pairs)}")
    if len(same_person_pairs) > 0:
        same_acc = len(same_correct) / len(same_person_pairs)
        print(f"  Correct: {len(same_correct)} ({same_acc*100:.2f}%)")
        print(f"  Missed: {len(same_person_pairs) - len(same_correct)}")
    
    print(f"\nDifferent Person Pairs:")
    print(f"  Total: {len(diff_person_pairs)}")
    if len(diff_person_pairs) > 0:
        diff_acc = len(diff_correct) / len(diff_person_pairs)
        print(f"  Correct: {len(diff_correct)} ({diff_acc*100:.2f}%)")
        print(f"  False Matches: {len(diff_person_pairs) - len(diff_correct)}")
    
    # Age gap breakdown
    print("\n" + "="*70)
    print("ACCURACY BY AGE GAP (Same-Person Pairs Only)")
    print("="*70)
    for category in ["small", "medium", "large", "xlarge"]:
        cat_pairs = [r for r in same_person_pairs if r.get("age_gap_category") == category]
        if cat_pairs:
            cat_correct = [r for r in cat_pairs if r["prediction_correct"]]
            cat_acc = len(cat_correct) / len(cat_pairs)
            avg_sim = sum(r["best_similarity"] for r in cat_pairs) / len(cat_pairs)
            print(f"  {category:8s}: {len(cat_correct):3d}/{len(cat_pairs):3d} = {cat_acc*100:5.2f}%  (avg similarity: {avg_sim:.3f})")
    
    # Image count analysis
    print("\n" + "="*70)
    print("ACCURACY BY IMAGE COUNT")
    print("="*70)
    for img_count in range(1, 11):
        pairs_with_count = [r for r in matchable 
                          if int(r.get("query_image_count", 0)) == img_count or 
                             int(r.get("candidate_image_count", 0)) == img_count]
        if pairs_with_count:
            correct = [r for r in pairs_with_count if r["prediction_correct"]]
            acc = len(correct) / len(pairs_with_count)
            print(f"  {img_count:2d} images: {len(correct):3d}/{len(pairs_with_count):3d} = {acc*100:5.2f}%")
    
    # Similarity distribution
    if HAS_PLOTTING:
        try:
            df = pd.DataFrame(matchable)
            df["is_same_person"] = df["is_same_person"].astype(bool)
            df["best_similarity"] = pd.to_numeric(df["best_similarity"], errors='coerce')
            
            same_sims = df[df["is_same_person"] == True]["best_similarity"].dropna()
            diff_sims = df[df["is_same_person"] == False]["best_similarity"].dropna()
            
            plt.figure(figsize=(14, 6))
            
            # Similarity distribution
            plt.subplot(1, 2, 1)
            plt.hist(same_sims, bins=30, alpha=0.7, label="Same Person", color='green', edgecolor='black')
            plt.hist(diff_sims, bins=30, alpha=0.7, label="Different Person", color='red', edgecolor='black')
            plt.axvline(0.30, color='orange', linestyle='--', linewidth=2, label="Threshold 0.30")
            plt.xlabel("Best Similarity Score")
            plt.ylabel("Frequency")
            plt.title("Similarity Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Age gap accuracy
            plt.subplot(1, 2, 2)
            age_gap_acc = []
            age_gap_labels = []
            for category in ["small", "medium", "large", "xlarge"]:
                cat_pairs = [r for r in same_person_pairs if r.get("age_gap_category") == category]
                if cat_pairs:
                    cat_correct = [r for r in cat_pairs if r["prediction_correct"]]
                    acc = len(cat_correct) / len(cat_pairs) if len(cat_pairs) > 0 else 0
                    age_gap_acc.append(acc * 100)
                    age_gap_labels.append(f"{category}\n(n={len(cat_pairs)})")
            
            if age_gap_acc:
                plt.bar(age_gap_labels, age_gap_acc, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
                plt.ylabel("Accuracy (%)")
                plt.title("Accuracy by Age Gap Category")
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            output_path = Path("tests/data/test_results_analysis.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.success(f"Plot saved to {output_path}")
            print(f"\n✅ Plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    analyze_results()

