"""
Phase 2 - Step B: Collect top-K search results for all Recall@K queries.

This script:
1. Loads the ground truth and ID mapping from Phase 2 Step A.
2. For each missing case, calls the search API with K ∈ {5, 10, 20, 50}.
3. Saves the aggregated results to tests/data/topk_results.json.

Usage:
    python tests/scripts/collect_topk_results.py

Environment variables:
    API_BASE_URL (default: http://localhost:8000)
    SEARCH_SLEEP_SECONDS (default: 0.3)  - delay between search requests
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import httpx

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tests.utils.ground_truth_loader import GroundTruth, TestCase, load_ground_truth

DATA_DIR = PROJECT_ROOT / "tests" / "data"
GROUND_TRUTH_PATH = DATA_DIR / "recall_at_k_ground_truth.json"
MAPPING_PATH = DATA_DIR / "id_mapping.json"
TOPK_RESULTS_PATH = DATA_DIR / "topk_results.json"

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
SEARCH_URL_TEMPLATE = f"{API_BASE}/api/v1/search/missing/{{case_id}}"
KS = [5, 10, 20, 50]

SEARCH_TIMEOUT = httpx.Timeout(120.0)
SLEEP_SECONDS = float(os.environ.get("SEARCH_SLEEP_SECONDS", "0.3"))


def ensure_dependencies():
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"ID mapping not found: {MAPPING_PATH}. "
            "Run tests/scripts/upload_test_cases.py first."
        )


def load_mapping() -> Dict[str, Dict[str, str]]:
    ensure_dependencies()
    data = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    return {
        "missing": data.get("missing", {}),
        "found": data.get("found", {}),
        "meta": data.get("meta", {}),
    }


def resolve_found_ids(case: TestCase, mapping_found: Dict[str, str]) -> List[str]:
    found_ids = []
    for name in case.positive_matches:
        found_id = mapping_found.get(name)
        if found_id:
            found_ids.append(found_id)
    return found_ids


async def fetch_topk(
    client: httpx.AsyncClient,
    case_id: str,
    limit: int,
) -> Dict[str, object]:
    url = SEARCH_URL_TEMPLATE.format(case_id=case_id)
    params = {"limit": limit}
    try:
        response = await client.get(url, params=params)
    except httpx.HTTPError as exc:
        return {
            "matched_ids": [],
            "scores": [],
            "count": 0,
            "error": str(exc),
        }
    if response.status_code != 200:
        return {
            "matched_ids": [],
            "scores": [],
            "count": 0,
            "error": f"HTTP {response.status_code}: {response.text}",
        }

    payload = response.json()
    matches = payload.get("matches", []) or []

    matched_ids = []
    scores = []
    for match in matches:
        metadata = match.get("metadata", {}) or {}
        matched_id = (
            metadata.get("found_id")
            or metadata.get("case_id")
            or match.get("id")
        )
        matched_ids.append(matched_id)
        scores.append(match.get("combined_score") or match.get("face_similarity") or 0.0)

    return {
        "matched_ids": matched_ids,
        "scores": scores,
        "count": len(matches),
    }


async def main():
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    mapping = load_mapping()
    missing_map = mapping["missing"]
    found_map = mapping["found"]

    topk_results: Dict[str, Dict[str, object]] = {}
    counts_per_k: Dict[int, List[int]] = defaultdict(list)
    queries_with_any_match = 0

    async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
        for case in ground_truth.test_cases:
            case_id = missing_map.get(case.query_id)
            if not case_id:
                print(f"⚠️  Skipping {case.query_id}: no case_id mapping.")
                continue

            ground_truth_found_ids = resolve_found_ids(case, found_map)
            per_case = {
                "case_id": case_id,
                "expected_in_topk": case.expected_in_topk,
                "ground_truth_positive": case.positive_matches,
                "ground_truth_found_ids": ground_truth_found_ids,
                "topk_results": {},
            }

            print(f"🔍 Searching {case.query_id} (case_id={case_id})...")
            any_match = False
            for k in KS:
                result = await fetch_topk(client, case_id, k)
                per_case["topk_results"][f"k={k}"] = result
                counts_per_k[k].append(result["count"])
                if result["count"] > 0:
                    any_match = True
                print(
                    f"   • K={k}: {result['count']} matches"
                    + (f" (error: {result['error']})" if "error" in result else "")
                )
                await asyncio.sleep(SLEEP_SECONDS)

            if any_match:
                queries_with_any_match += 1

            topk_results[case.query_id] = per_case

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TOPK_RESULTS_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat(),
                "results": topk_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("═══════════════════════════════════════════════")
    print(f"✅ Saved top-K results to {TOPK_RESULTS_PATH}")
    print(f"✅ Queries with at least one match: {queries_with_any_match}/{len(topk_results)}")
    for k in KS:
        counts = counts_per_k.get(k, [])
        avg = sum(counts) / len(counts) if counts else 0.0
        print(f"   • Average results for K={k}: {avg:.2f}")
    print("═══════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())

