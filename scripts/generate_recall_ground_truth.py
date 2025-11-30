"""
Utilities for building Recall@K ground truth dataset.

Phase 1 focuses on:
1. Scanning FGNET dataset metadata.
2. Generating balanced multi-image aware test cases.
3. Saving ground truth JSON with rich statistics.

This module exposes reusable functions so other scripts/tests can import them:
    - scan_dataset()
    - generate_test_cases()
    - save_ground_truth()

It can also be executed directly from the command line to run the full pipeline:

    python scripts/generate_recall_ground_truth.py --run-all
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "FGNET_organized"
GROUND_TRUTH_PATH = PROJECT_ROOT / "tests" / "data" / "recall_at_k_ground_truth.json"

AGE_PATTERN = re.compile(r"age_(\d+)", re.IGNORECASE)

TEST_TYPE_CONFIG = {
    "single-single": {"query": 1, "positive": (1, 1)},  # exactly 1
    "single-multi": {"query": 1, "positive": (2, 3)},
    "multi-single": {"query": (2, 3), "positive": (1, 1)},
    "multi-multi": {"query": (2, 3), "positive": (2, 3)},
}

AGE_GAP_BUCKETS = {
    "small": (0, 5),
    "medium": (6, 10),
    "large": (11, 200),
}


@dataclass
class PersonImage:
    age: int
    path: Path


@dataclass
class PersonInfo:
    person_id: str
    images: List[PersonImage] = field(default_factory=list)

    def sorted_images(self) -> List[PersonImage]:
        return sorted(self.images, key=lambda img: img.age)

    @property
    def image_count(self) -> int:
        return len(self.images)


def parse_age_from_filename(filename: str) -> Optional[int]:
    match = AGE_PATTERN.search(filename)
    if not match:
        return None
    return int(match.group(1))


def scan_dataset(base_path: Path = DATASET_ROOT) -> Dict[str, object]:
    """
    Scan FGNET dataset and build structured metadata.

    Returns:
        dict with keys:
            persons: List[PersonInfo]
            stats: Dict[str, object]
    """
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")

    persons: List[PersonInfo] = []
    total_images = 0
    distribution_counter = Counter()

    for person_dir in sorted(base_path.glob("person_*")):
        if not person_dir.is_dir():
            continue

        images: List[PersonImage] = []
        for image_file in sorted(person_dir.glob("*.jpg")):
            age = parse_age_from_filename(image_file.name)
            if age is None:
                continue
            images.append(PersonImage(age=age, path=image_file.relative_to(PROJECT_ROOT)))

        if not images:
            continue

        person_info = PersonInfo(person_id=person_dir.name, images=images)
        persons.append(person_info)
        total_images += len(images)

        if len(images) == 1:
            distribution_counter["1_image"] += 1
        elif len(images) == 2:
            distribution_counter["2_images"] += 1
        elif len(images) == 3:
            distribution_counter["3_images"] += 1
        else:
            distribution_counter["4+_images"] += 1

    stats = {
        "dataset_path": str(base_path.relative_to(PROJECT_ROOT)),
        "total_persons": len(persons),
        "total_images": total_images,
        "distribution": distribution_counter,
    }

    return {"persons": persons, "stats": stats}


def _ensure_range(value_or_tuple: Sequence[int] | int, default: int) -> Tuple[int, int]:
    if isinstance(value_or_tuple, int):
        return value_or_tuple, value_or_tuple
    if isinstance(value_or_tuple, Sequence) and len(value_or_tuple) == 2:
        return int(value_or_tuple[0]), int(value_or_tuple[1])
    return default, default


def _select_images_for_case(
    person: PersonInfo,
    query_count: Tuple[int, int],
    positive_count: Tuple[int, int],
    gap_bucket: str,
) -> Optional[Tuple[List[PersonImage], List[PersonImage], int]]:
    images = person.sorted_images()
    q_min, q_max = query_count
    p_min, p_max = positive_count

    bucket_min, bucket_max = AGE_GAP_BUCKETS[gap_bucket]

    possible_q_counts = range(q_min, q_max + 1)
    possible_p_counts = range(p_min, p_max + 1)

    indices = list(range(len(images)))

    for q in possible_q_counts:
        if q >= len(images):
            continue
        for query_indices in combinations(indices, q):
            remaining = [idx for idx in indices if idx not in query_indices]
            for p in possible_p_counts:
                if p > len(remaining) or p == 0:
                    continue
                for positive_indices in combinations(remaining, p):
                    query_imgs = [images[i] for i in query_indices]
                    positive_imgs = [images[i] for i in positive_indices]
                    gap = positive_imgs[-1].age - query_imgs[0].age
                    if bucket_min <= gap <= bucket_max:
                        return query_imgs, positive_imgs, gap
    return None


def generate_test_cases(
    dataset_info: Dict[str, object],
    seed: int = 42,
    target_positive: int = 40,
    target_negative: int = 10,
) -> List[Dict[str, object]]:
    """
    Generate balanced Recall@K test cases.
    """
    random.seed(seed)

    persons: List[PersonInfo] = list(dataset_info["persons"])  # type: ignore
    random.shuffle(persons)

    eligible = {
        "single-single": [p for p in persons if p.image_count >= 2],
        "single-multi": [p for p in persons if p.image_count >= 3],
        "multi-single": [p for p in persons if p.image_count >= 3],
        "multi-multi": [p for p in persons if p.image_count >= 4],
        "negative": persons,
    }

    plan = {
        "single-single": 10,
        "single-multi": 10,
        "multi-single": 10,
        "multi-multi": 10,
    }

    gap_sequence = ["small"] * 15 + ["medium"] * 15 + ["large"] * 10
    random.shuffle(gap_sequence)

    gap_iter = iter(gap_sequence)
    test_cases: List[Dict[str, object]] = []
    used_person_ids: set[str] = set()

    case_counter = 1

    # Positive cases
    for test_type, count in plan.items():
        for _ in range(count):
            gap_bucket = next(gap_iter, "large")
            for person in eligible[test_type]:
                if person.person_id in used_person_ids:
                    continue
                combo = _select_images_for_case(
                    person,
                    _ensure_range(TEST_TYPE_CONFIG[test_type]["query"], 1),
                    _ensure_range(TEST_TYPE_CONFIG[test_type]["positive"], 1),
                    gap_bucket,
                )
                if combo:
                    query_imgs, positive_imgs, gap = combo
                    test_cases.append(
                        _build_test_case_dict(
                            case_counter,
                            test_type,
                            person.person_id,
                            query_imgs,
                            positive_imgs,
                            gap,
                            expected_in_topk=True,
                        )
                    )
                    case_counter += 1
                    used_person_ids.add(person.person_id)
                    break
            else:
                raise RuntimeError(
                    f"Unable to create {test_type} case for gap bucket '{gap_bucket}'. "
                    "Consider adjusting dataset requirements."
                )

    # Negative cases
    negative_needed = target_negative
    for person in persons:
        if negative_needed == 0:
            break
        if person.person_id in used_person_ids and person.image_count > 1:
            continue
        query_imgs = [person.sorted_images()[0]]
        test_cases.append(
            _build_test_case_dict(
                case_counter,
                "single-single",
                person.person_id,
                query_imgs,
                positive_imgs=[],
                gap=0,
                expected_in_topk=False,
            )
        )
        case_counter += 1
        negative_needed -= 1

    if negative_needed > 0:
        raise RuntimeError("Insufficient persons to create negative test cases.")

    return test_cases


def _build_test_case_dict(
    case_index: int,
    test_type: str,
    person_id: str,
    query_imgs: List[PersonImage],
    positive_imgs: List[PersonImage],
    gap: int,
    expected_in_topk: bool,
) -> Dict[str, object]:
    query_paths = [str(img.path).replace("\\", "/") for img in query_imgs]
    query_ages = [img.age for img in query_imgs]

    positive_paths = [str(img.path).replace("\\", "/") for img in positive_imgs]
    positive_ages = [img.age for img in positive_imgs]

    test_case = {
        "query_id": f"MISS_{case_index:03d}",
        "query_person_name": person_id,
        "query_images": query_paths,
        "query_ages": query_ages,
        "num_query_images": len(query_paths),
        "positive_matches": [person_id] if expected_in_topk else [],
        "positive_person_name": person_id if expected_in_topk else None,
        "positive_images": positive_paths,
        "positive_ages": positive_ages,
        "num_positive_images": len(positive_paths),
        "expected_in_topk": expected_in_topk,
        "test_type": test_type,
        "age_gap": gap,
        "notes": _generate_notes(test_type, gap, expected_in_topk),
    }

    if not expected_in_topk:
        test_case["positive_person_name"] = None

    return test_case


def _generate_notes(test_type: str, gap: int, expected: bool) -> str:
    if not expected:
        return "Negative control case - no matching person in database snapshot."
    descriptors = {
        "single-single": "Single query vs single candidate",
        "single-multi": "Single query vs multi-image candidate",
        "multi-single": "Multi-image query vs single candidate",
        "multi-multi": "Full multi-image aggregation on both sides",
    }
    gap_label = (
        "small"
        if gap <= 5
        else "medium"
        if gap <= 10
        else "large"
    )
    return f"{descriptors.get(test_type, 'Match case')} with {gap_label} age gap ({gap} years)."


def calculate_statistics(test_cases: List[Dict[str, object]]) -> Dict[str, object]:
    total_cases = len(test_cases)
    positive_cases = sum(1 for case in test_cases if case["expected_in_topk"])
    negative_cases = total_cases - positive_cases

    type_counter = Counter(case["test_type"] for case in test_cases if case["expected_in_topk"])
    gap_counter = Counter()

    total_query_images = 0
    total_positive_images = 0
    min_gap = float("inf")
    max_gap = 0

    for case in test_cases:
        total_query_images += case["num_query_images"]
        total_positive_images += case["num_positive_images"]
        gap = case.get("age_gap", 0) or 0
        if case["expected_in_topk"]:
            if gap <= 5:
                gap_counter["small (0-5y)"] += 1
            elif gap <= 10:
                gap_counter["medium (6-10y)"] += 1
            else:
                gap_counter["large (11+y)"] += 1
            min_gap = min(min_gap, gap)
            max_gap = max(max_gap, gap)

    avg_query = total_query_images / total_cases if total_cases else 0
    avg_positive = total_positive_images / positive_cases if positive_cases else 0

    return {
        "total_cases": total_cases,
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "test_type_distribution": dict(type_counter),
        "age_gap_distribution": dict(gap_counter),
        "avg_images_per_query": round(avg_query, 2),
        "avg_images_per_found": round(avg_positive, 2),
        "min_age_gap": min_gap if min_gap != float("inf") else 0,
        "max_age_gap": max_gap,
    }


def validate_test_cases(test_cases: List[Dict[str, object]]) -> None:
    seen_ids = set()
    for case in test_cases:
        query_id = case["query_id"]
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id detected: {query_id}")
        seen_ids.add(query_id)

        for img_path in case["query_images"]:
            if not (PROJECT_ROOT / img_path).exists():
                raise FileNotFoundError(f"Query image not found: {img_path}")

        for img_path in case["positive_images"]:
            if not (PROJECT_ROOT / img_path).exists():
                raise FileNotFoundError(f"Positive image not found: {img_path}")

        if case["expected_in_topk"]:
            if not case["positive_matches"]:
                raise ValueError(f"Positive case {query_id} missing positive_matches.")
        else:
            if case["positive_matches"]:
                raise ValueError(f"Negative case {query_id} should not have positive matches.")


def save_ground_truth(test_cases: List[Dict[str, object]], output_path: Path = GROUND_TRUTH_PATH) -> None:
    validate_test_cases(test_cases)
    stats = calculate_statistics(test_cases)

    data = {
        "version": "1.0",
        "description": "Ground truth for Recall@K evaluation - Multi-image aware",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "FGNET Organized",
        "evaluation_config": {
            "k_values": [5, 10, 20, 50],
            "supports_multi_image": True,
        },
        "statistics": stats,
        "test_cases": test_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Ground truth saved to {output_path}")


def print_dataset_summary(scan_result: Dict[str, object]) -> None:
    stats = scan_result["stats"]
    distribution = stats["distribution"]
    print("═══════════════════════════════════════════════")
    print(f"✅ Scanned {stats['dataset_path']}")
    print(f"✅ Total persons: {stats['total_persons']}")
    print(f"✅ Total images: {stats['total_images']}")
    print("✅ Distribution:")
    for key in ["1_image", "2_images", "3_images", "4+_images"]:
        print(f"   - {key}: {distribution.get(key, 0)}")
    print("═══════════════════════════════════════════════")


def run_full_pipeline() -> None:
    scan_result = scan_dataset()
    print_dataset_summary(scan_result)
    test_cases = generate_test_cases(scan_result)
    print(f"✅ Generated {len(test_cases)} test cases.")
    save_ground_truth(test_cases)


def main():
    parser = argparse.ArgumentParser(description="Recall@K ground truth generator")
    parser.add_argument("--scan", action="store_true", help="Scan dataset only")
    parser.add_argument("--generate", action="store_true", help="Generate test cases and save ground truth")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline (scan + generate + save)")
    args = parser.parse_args()

    if args.run_all:
        run_full_pipeline()
        return

    if args.scan:
        scan_result = scan_dataset()
        print_dataset_summary(scan_result)

    if args.generate:
        scan_result = scan_dataset()
        test_cases = generate_test_cases(scan_result)
        save_ground_truth(test_cases)
        print("✅ Generation complete.")

    if not any([args.scan, args.generate, args.run_all]):
        parser.print_help()


if __name__ == "__main__":
    main()
