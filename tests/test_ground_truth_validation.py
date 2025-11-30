"""
Validation tests for Recall@K ground truth dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.ground_truth_loader import (
    DEFAULT_GROUND_TRUTH_PATH,
    GroundTruth,
    TestCase,
    get_negative_cases,
    get_positive_cases,
    print_summary,
    load_ground_truth,
)


@pytest.fixture(scope="session")
def ground_truth() -> GroundTruth:
    return load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)


def test_ground_truth_file_exists():
    assert DEFAULT_GROUND_TRUTH_PATH.exists(), (
        f"Ground truth file missing: {DEFAULT_GROUND_TRUTH_PATH}"
    )


def test_ground_truth_loads_successfully(ground_truth: GroundTruth):
    assert ground_truth.test_cases, "Ground truth contains no test cases."


def test_minimum_test_cases(ground_truth: GroundTruth):
    assert ground_truth.statistics.total_cases >= 50, (
        "Ground truth must include at least 50 cases."
    )


def test_all_query_images_exist(ground_truth: GroundTruth):
    for case in ground_truth.test_cases:
        for path_str in case.query_images:
            assert (Path.cwd() / path_str).exists(), f"Missing query image: {path_str}"


def test_all_positive_images_exist(ground_truth: GroundTruth):
    for case in get_positive_cases(ground_truth):
        for path_str in case.positive_images:
            assert (Path.cwd() / path_str).exists(), f"Missing positive image: {path_str}"


def test_positive_cases_valid(ground_truth: GroundTruth):
    for case in get_positive_cases(ground_truth):
        assert case.expected_in_topk
        assert case.positive_matches, f"{case.query_id} missing positive matches."
        assert case.positive_person_name, f"{case.query_id} missing positive person name."


def test_negative_cases_valid(ground_truth: GroundTruth):
    for case in get_negative_cases(ground_truth):
        assert not case.expected_in_topk
        assert not case.positive_matches
        assert case.positive_person_name is None
        assert case.num_positive_images == 0


def test_query_ids_unique(ground_truth: GroundTruth):
    ids = [case.query_id for case in ground_truth.test_cases]
    assert len(ids) == len(set(ids)), "Duplicate query IDs detected."


def test_statistics_match_actual(ground_truth: GroundTruth):
    stats = ground_truth.statistics
    actual_total = len(ground_truth.test_cases)
    actual_positive = len(get_positive_cases(ground_truth))
    actual_negative = len(get_negative_cases(ground_truth))
    assert stats.total_cases == actual_total
    assert stats.positive_cases == actual_positive
    assert stats.negative_cases == actual_negative


def test_test_type_distribution(ground_truth: GroundTruth):
    counts = {}
    for case in get_positive_cases(ground_truth):
        counts[case.test_type] = counts.get(case.test_type, 0) + 1
    assert counts == ground_truth.statistics.test_type_distribution


def test_age_gap_reasonable(ground_truth: GroundTruth):
    for case in ground_truth.test_cases:
        assert 0 <= case.age_gap <= 120, f"Unreasonable age gap in {case.query_id}"


def test_multi_image_cases_have_multiple_images(ground_truth: GroundTruth):
    for case in ground_truth.test_cases:
        if case.test_type == "single-single":
            assert case.num_query_images == 1
            if case.expected_in_topk:
                assert case.num_positive_images == 1
        elif case.test_type == "single-multi":
            assert case.num_query_images == 1
            if case.expected_in_topk:
                assert case.num_positive_images >= 2
        elif case.test_type == "multi-single":
            assert case.num_query_images >= 2
            if case.expected_in_topk:
                assert case.num_positive_images == 1
        elif case.test_type == "multi-multi":
            assert case.num_query_images >= 2
            if case.expected_in_topk:
                assert case.num_positive_images >= 2


def test_print_full_summary(ground_truth: GroundTruth, capsys):
    print_summary(ground_truth)
    captured = capsys.readouterr()
    assert "Ground Truth Version" in captured.out

