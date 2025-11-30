#!/usr/bin/env python3
"""
Phase 3: Calculate Recall@K Metrics & Visualization

Computes Recall@K, MRR@K, Precision@K from top-K search results
and generates comprehensive evaluation reports with visualizations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "tests" / "data"
SCRIPTS_DIR = PROJECT_ROOT / "tests" / "scripts"

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# K values to evaluate
K_VALUES = [5, 10, 20, 50]

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_ground_truth() -> Dict:
    """Load ground truth test cases."""
    gt_path = DATA_DIR / "recall_at_k_ground_truth.json"
    print(f"📂 Loading ground truth from: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data['test_cases'])} test cases")
    return data


def load_id_mapping() -> Dict:
    """Load ID mapping between original and API-generated IDs."""
    mapping_path = DATA_DIR / "id_mapping.json"
    print(f"📂 Loading ID mapping from: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data['missing'])} missing IDs, {len(data['found'])} found IDs")
    return data


def load_topk_results() -> Dict:
    """Load top-K search results."""
    results_path = DATA_DIR / "topk_results.json"
    print(f"📂 Loading top-K results from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", data)
    print(f"✅ Loaded {len(results)} query results")
    return results


def get_positive_cases(gt_data: Dict) -> List[Dict]:
    """Get all positive test cases (expected_in_topk=True)."""
    return [case for case in gt_data["test_cases"] if case.get("expected_in_topk", False)]


def compute_recall_at_k(
    topk_matched_ids: List[str],
    ground_truth_found_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@K for a single query.
    
    Recall@K = 1 if any ground_truth_found_id is in top-K, else 0.
    """
    if not ground_truth_found_ids:
        return 0.0
    
    # Check if any ground truth ID appears in top-K
    topk_set = set(topk_matched_ids[:k])
    gt_set = set(ground_truth_found_ids)
    
    if gt_set & topk_set:  # Intersection exists
        return 1.0
    return 0.0


def compute_mrr_at_k(
    topk_matched_ids: List[str],
    ground_truth_found_ids: List[str],
    k: int
) -> float:
    """
    Compute MRR@K (Mean Reciprocal Rank) for a single query.
    
    MRR@K = 1/rank of first ground_truth match if rank <= K, else 0.
    """
    if not ground_truth_found_ids:
        return 0.0
    
    # Find the first ground truth match and its rank
    for gt_id in ground_truth_found_ids:
        try:
            rank = topk_matched_ids.index(gt_id) + 1  # 1-indexed
            if rank <= k:
                return 1.0 / rank
        except ValueError:
            continue
    
    return 0.0


def compute_precision_at_k(
    topk_matched_ids: List[str],
    ground_truth_found_ids: List[str],
    k: int
) -> float:
    """
    Compute Precision@K for a single query.
    
    Precision@K = (number of true positives in top-K) / K
    """
    if k == 0:
        return 0.0
    
    topk_set = set(topk_matched_ids[:k])
    gt_set = set(ground_truth_found_ids)
    
    true_positives = len(gt_set & topk_set)
    return true_positives / k


def compute_metrics_for_k(
    k: int,
    gt_data: Dict,
    topk_results: Dict
) -> Tuple[Dict, List[float], List[float], List[float]]:
    """
    Compute all metrics for a specific K value.
    
    Returns:
        - metrics_dict: aggregated metrics
        - recall_list: per-query recall values
        - mrr_list: per-query MRR values
        - precision_list: per-query precision values
    """
    positive_cases = get_positive_cases(gt_data)
    recall_list = []
    mrr_list = []
    precision_list = []
    
    for case in positive_cases:
        query_id = case["query_id"]
        
        if query_id not in topk_results:
            print(f"⚠️  Warning: Query {query_id} not found in results")
            continue
        
        result = topk_results[query_id]
        k_key = f"k={k}"
        
        if k_key not in result["topk_results"]:
            print(f"⚠️  Warning: {k_key} not found for query {query_id}")
            continue
        
        topk_data = result["topk_results"][k_key]
        matched_ids = topk_data["matched_ids"]
        ground_truth_found_ids = result.get("ground_truth_found_ids", [])
        
        # Compute metrics
        recall = compute_recall_at_k(matched_ids, ground_truth_found_ids, k)
        mrr = compute_mrr_at_k(matched_ids, ground_truth_found_ids, k)
        precision = compute_precision_at_k(matched_ids, ground_truth_found_ids, k)
        
        recall_list.append(recall)
        mrr_list.append(mrr)
        precision_list.append(precision)
    
    # Aggregate metrics
    metrics = {
        "recall": np.mean(recall_list) if recall_list else 0.0,
        "mrr": np.mean(mrr_list) if mrr_list else 0.0,
        "precision": np.mean(precision_list) if precision_list else 0.0,
        "hits": sum(1 for r in recall_list if r > 0),
        "total_queries": len(recall_list)
    }
    
    return metrics, recall_list, mrr_list, precision_list


def compute_test_type_breakdown(
    gt_data: Dict,
    topk_results: Dict
) -> Dict[str, Dict[int, Dict]]:
    """
    Compute metrics breakdown by test type.
    
    Returns:
        {
            "single-single": {5: {...}, 10: {...}, ...},
            "single-multi": {...},
            ...
        }
    """
    breakdown = defaultdict(lambda: defaultdict(dict))
    positive_cases = get_positive_cases(gt_data)
    
    # Group cases by test type
    cases_by_type = defaultdict(list)
    for case in positive_cases:
        test_type = case.get("test_type", "unknown")
        cases_by_type[test_type].append(case)
    
    # Compute metrics for each test type and K value
    for test_type, cases in cases_by_type.items():
        for k in K_VALUES:
            recall_list = []
            mrr_list = []
            precision_list = []
            
            for case in cases:
                query_id = case["query_id"]
                
                if query_id not in topk_results:
                    continue
                
                result = topk_results[query_id]
                k_key = f"k={k}"
                
                if k_key not in result["topk_results"]:
                    continue
                
                topk_data = result["topk_results"][k_key]
                matched_ids = topk_data["matched_ids"]
                ground_truth_found_ids = result.get("ground_truth_found_ids", [])
                
                recall = compute_recall_at_k(matched_ids, ground_truth_found_ids, k)
                mrr = compute_mrr_at_k(matched_ids, ground_truth_found_ids, k)
                precision = compute_precision_at_k(matched_ids, ground_truth_found_ids, k)
                
                recall_list.append(recall)
                mrr_list.append(mrr)
                precision_list.append(precision)
            
            breakdown[test_type][k] = {
                "recall": np.mean(recall_list) if recall_list else 0.0,
                "mrr": np.mean(mrr_list) if mrr_list else 0.0,
                "precision": np.mean(precision_list) if precision_list else 0.0,
                "hits": sum(1 for r in recall_list if r > 0),
                "total_cases": len(recall_list)
            }
    
    return breakdown


def plot_recall_curve(metrics: Dict[int, Dict], output_path: Path):
    """Plot Recall@K curve."""
    ks = sorted(metrics.keys())
    recalls = [metrics[k]["recall"] for k in ks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, recalls, marker='o', linewidth=3, markersize=10, color='#2E86AB')
    plt.title("Recall@K - Face Matching System", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("K", fontsize=14, fontweight='bold')
    plt.ylabel("Recall@K", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 1.05])
    plt.xlim([0, max(ks) + 5])
    
    # Add value labels on points
    for k, recall in zip(ks, recalls):
        plt.annotate(f'{recall:.3f}', (k, recall), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_mrr_curve(metrics: Dict[int, Dict], output_path: Path):
    """Plot MRR@K curve."""
    ks = sorted(metrics.keys())
    mrrs = [metrics[k]["mrr"] for k in ks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, mrrs, marker='s', linewidth=3, markersize=10, color='#A23B72')
    plt.title("MRR@K - Face Matching System", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("K", fontsize=14, fontweight='bold')
    plt.ylabel("MRR@K", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 1.05])
    plt.xlim([0, max(ks) + 5])
    
    # Add value labels
    for k, mrr in zip(ks, mrrs):
        plt.annotate(f'{mrr:.3f}', (k, mrr), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_precision_curve(metrics: Dict[int, Dict], output_path: Path):
    """Plot Precision@K curve."""
    ks = sorted(metrics.keys())
    precisions = [metrics[k]["precision"] for k in ks]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, precisions, marker='^', linewidth=3, markersize=10, color='#F18F01')
    plt.title("Precision@K - Face Matching System", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("K", fontsize=14, fontweight='bold')
    plt.ylabel("Precision@K", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, max(precisions) * 1.2])
    plt.xlim([0, max(ks) + 5])
    
    # Add value labels
    for k, precision in zip(ks, precisions):
        plt.annotate(f'{precision:.3f}', (k, precision), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def plot_test_type_breakdown(breakdown: Dict[str, Dict[int, Dict]], output_path: Path):
    """Plot Recall@K by test type (grouped bar chart)."""
    test_types = sorted(breakdown.keys())
    ks = K_VALUES
    
    # Prepare data
    data = []
    for test_type in test_types:
        recalls = [breakdown[test_type][k]["recall"] for k in ks]
        data.append(recalls)
    
    # Create grouped bar chart
    x = np.arange(len(ks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (test_type, recalls) in enumerate(zip(test_types, data)):
        offset = (i - len(test_types) / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=test_type, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, recall in zip(bars, recalls):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{recall:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('K', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall@K', fontsize=14, fontweight='bold')
    ax.set_title('Recall@K by Test Type', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in ks])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")


def print_overall_metrics_table(metrics: Dict[int, Dict]):
    """Print overall metrics table."""
    print("\n" + "="*70)
    print("📊 OVERALL METRICS")
    print("="*70)
    
    table_data = []
    for k in K_VALUES:
        m = metrics[k]
        table_data.append([
            k,
            f"{m['recall']:.3f}",
            f"{m['mrr']:.3f}",
            f"{m['precision']:.3f}",
            f"{m['hits']}/{m['total_queries']}"
        ])
    
    headers = ["K", "Recall@K", "MRR@K", "P@K", "Hits"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))


def print_test_type_table(breakdown: Dict[str, Dict[int, Dict]]):
    """Print test type breakdown table."""
    print("\n" + "="*70)
    print("📊 RECALL@K BY TEST TYPE")
    print("="*70)
    
    test_types = sorted(breakdown.keys())
    table_data = []
    
    for test_type in test_types:
        row = [test_type]
        for k in K_VALUES:
            recall = breakdown[test_type][k]["recall"]
            row.append(f"{recall:.3f}")
        table_data.append(row)
    
    headers = ["Test Type"] + [f"R@{k}" for k in K_VALUES]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))


def save_json_report(metrics: Dict[int, Dict], breakdown: Dict[str, Dict[int, Dict]], output_path: Path):
    """Save comprehensive JSON report."""
    report = {
        "version": "1.0",
        "generated_at": str(Path(__file__).stat().st_mtime),
        "k_values": K_VALUES,
        "overall_metrics": {
            str(k): {
                "recall": float(metrics[k]["recall"]),
                "mrr": float(metrics[k]["mrr"]),
                "precision": float(metrics[k]["precision"]),
                "hits": int(metrics[k]["hits"]),
                "total_queries": int(metrics[k]["total_queries"])
            }
            for k in K_VALUES
        },
        "test_type_breakdown": {
            test_type: {
                str(k): {
                    "recall": float(breakdown[test_type][k]["recall"]),
                    "mrr": float(breakdown[test_type][k]["mrr"]),
                    "precision": float(breakdown[test_type][k]["precision"]),
                    "hits": int(breakdown[test_type][k]["hits"]),
                    "total_cases": int(breakdown[test_type][k]["total_cases"])
                }
                for k in K_VALUES
            }
            for test_type in sorted(breakdown.keys())
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved: {output_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("🔥 PHASE 3: CALCULATE RECALL@K METRICS & VISUALIZATION")
    print("="*70)
    print()
    
    # Load data
    gt_data = load_ground_truth()
    id_mapping = load_id_mapping()  # Not used in calculations but loaded for completeness
    topk_results = load_topk_results()
    
    print("\n" + "="*70)
    print("📈 COMPUTING METRICS")
    print("="*70)
    
    # Compute overall metrics for each K
    metrics = {}
    for k in K_VALUES:
        print(f"Computing metrics for K={k}...")
        metrics[k], _, _, _ = compute_metrics_for_k(k, gt_data, topk_results)
    
    # Compute test type breakdown
    print("\nComputing test type breakdown...")
    breakdown = compute_test_type_breakdown(gt_data, topk_results)
    
    # Print tables
    print_overall_metrics_table(metrics)
    print_test_type_table(breakdown)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_recall_curve(metrics, DATA_DIR / "recall_at_k_curve.png")
    plot_mrr_curve(metrics, DATA_DIR / "mrr_at_k_curve.png")
    plot_precision_curve(metrics, DATA_DIR / "precision_at_k_curve.png")
    plot_test_type_breakdown(breakdown, DATA_DIR / "recall_by_test_type.png")
    
    # Save JSON report
    print("\n" + "="*70)
    print("💾 SAVING REPORTS")
    print("="*70)
    
    save_json_report(metrics, breakdown, DATA_DIR / "recall_at_k_report.json")
    
    print("\n" + "="*70)
    print("✅ PHASE 3 COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"  - {DATA_DIR / 'recall_at_k_report.json'}")
    print(f"  - {DATA_DIR / 'recall_at_k_curve.png'}")
    print(f"  - {DATA_DIR / 'mrr_at_k_curve.png'}")
    print(f"  - {DATA_DIR / 'precision_at_k_curve.png'}")
    print(f"  - {DATA_DIR / 'recall_by_test_type.png'}")


if __name__ == "__main__":
    main()
