"""
Ground truth loader utilities for Recall@K evaluation.

Provides strongly-typed access to the JSON ground truth file generated in Phase 1.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH_PATH = PROJECT_ROOT / "tests" / "data" / "recall_at_k_ground_truth.json"


class TestCase(BaseModel):
    query_id: str
    query_person_name: str
    query_images: List[str]
    query_ages: List[int]
    num_query_images: int
    expected_in_topk: bool
    positive_matches: List[str]
    positive_person_name: Optional[str] = Field(default=None)
    positive_images: List[str]
    positive_ages: List[int]
    num_positive_images: int
    test_type: str
    age_gap: int = Field(ge=0)
    notes: Optional[str] = None

    @field_validator("query_images", "positive_images")
    @classmethod
    def validate_image_paths(cls, values: List[str]) -> List[str]:
        for path_str in values:
            full_path = PROJECT_ROOT / path_str
            if not full_path.exists():
                raise ValueError(f"Image not found: {path_str}")
        return values

    @field_validator("num_query_images")
    @classmethod
    def validate_query_count(cls, value: int, info) -> int:
        images = info.data.get("query_images", [])
        if value != len(images):
            raise ValueError(
                f"num_query_images ({value}) does not match query_images length ({len(images)})"
            )
        return value

    @field_validator("num_positive_images")
    @classmethod
    def validate_positive_count(cls, value: int, info) -> int:
        images = info.data.get("positive_images", [])
        if value != len(images):
            raise ValueError(
                f"num_positive_images ({value}) does not match positive_images length ({len(images)})"
            )
        return value

    @field_validator("positive_matches")
    @classmethod
    def validate_positive_expectation(cls, values: List[str], info) -> List[str]:
        expected = info.data.get("expected_in_topk", False)
        if expected and not values:
            raise ValueError("Positive cases must include at least one positive match.")
        if not expected and values:
            raise ValueError("Negative cases must not include positive matches.")
        return values


TestCase.__test__ = False  # Prevent pytest from collecting this Pydantic model.


class Statistics(BaseModel):
    total_cases: int
    positive_cases: int
    negative_cases: int
    test_type_distribution: Dict[str, int]
    age_gap_distribution: Dict[str, int]
    avg_images_per_query: float
    avg_images_per_found: float
    min_age_gap: int
    max_age_gap: int

    @model_validator(mode="after")
    def validate_counts(self):
        if self.total_cases != self.positive_cases + self.negative_cases:
            raise ValueError("total_cases must equal positive_cases + negative_cases.")
        return self


class GroundTruth(BaseModel):
    version: str
    description: str
    created_at: datetime
    dataset: str
    evaluation_config: Dict[str, object]
    statistics: Statistics
    test_cases: List[TestCase]

    @model_validator(mode="after")
    def validate_case_count(self):
        if len(self.test_cases) != self.statistics.total_cases:
            raise ValueError(
                f"statistics.total_cases ({self.statistics.total_cases}) "
                f"does not match actual test case count ({len(self.test_cases)})."
            )
        ids = set()
        for case in self.test_cases:
            if case.query_id in ids:
                raise ValueError(f"Duplicate query_id detected: {case.query_id}")
            ids.add(case.query_id)
        return self


def load_ground_truth(path: Path | str = DEFAULT_GROUND_TRUTH_PATH) -> GroundTruth:
    """
    Load and validate the recall@K ground truth JSON file.
    """
    gt_path = Path(path)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    raw_data = gt_path.read_text(encoding="utf-8")
    try:
        return GroundTruth.model_validate_json(raw_data)
    except ValidationError as exc:
        raise ValueError(f"Ground truth validation failed: {exc}") from exc


def get_positive_cases(ground_truth: GroundTruth) -> List[TestCase]:
    return [case for case in ground_truth.test_cases if case.expected_in_topk]


def get_negative_cases(ground_truth: GroundTruth) -> List[TestCase]:
    return [case for case in ground_truth.test_cases if not case.expected_in_topk]


def get_cases_by_type(ground_truth: GroundTruth, test_type: str) -> List[TestCase]:
    return [case for case in ground_truth.test_cases if case.test_type == test_type]


def get_cases_by_age_gap(
    ground_truth: GroundTruth,
    *,
    min_gap: Optional[int] = None,
    max_gap: Optional[int] = None,
) -> List[TestCase]:
    def within(case: TestCase) -> bool:
        gap = case.age_gap
        if min_gap is not None and gap < min_gap:
            return False
        if max_gap is not None and gap > max_gap:
            return False
        return True

    return [case for case in ground_truth.test_cases if within(case)]


def print_summary(ground_truth: GroundTruth) -> None:
    stats = ground_truth.statistics
    print("═══════════════════════════════════════════════")
    print(f"Ground Truth Version: {ground_truth.version}")
    print(f"Created at: {ground_truth.created_at.isoformat()}")
    print(f"Dataset: {ground_truth.dataset}")
    print(f"Total Cases: {stats.total_cases} "
          f"(Positives: {stats.positive_cases} / Negatives: {stats.negative_cases})")
    print("Test Type Distribution:")
    for test_type, count in stats.test_type_distribution.items():
        print(f"  - {test_type}: {count}")
    print("Age Gap Distribution:")
    for bucket, count in stats.age_gap_distribution.items():
        print(f"  - {bucket}: {count}")
    print(
        f"Average images → Query: {stats.avg_images_per_query}, "
        f"Found: {stats.avg_images_per_found}"
    )
    print(f"Age gap range: {stats.min_age_gap} - {stats.max_age_gap} years")
    print("═══════════════════════════════════════════════")


__all__ = [
    "GroundTruth",
    "TestCase",
    "Statistics",
    "load_ground_truth",
    "get_positive_cases",
    "get_negative_cases",
    "get_cases_by_type",
    "get_cases_by_age_gap",
    "print_summary",
]

