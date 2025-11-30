"""
Phase 2 - Step A: Upload ground truth test cases to the API.

This script:
1. Loads the Recall@K ground truth JSON.
2. Uploads each test case as a missing person (multi-image batch endpoint).
3. Uploads corresponding found persons for positive cases.
4. Stores the mapping between query IDs / person names and the API-generated IDs.

Usage:
    python tests/scripts/upload_test_cases.py

Environment variables:
    API_BASE_URL (default: http://localhost:8000)
    UPLOAD_SLEEP_SECONDS (default: 0.5)  - delay between missing uploads
    UPLOAD_FOUND_SLEEP_SECONDS (default: 0.5) - delay between found uploads
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tests.utils.ground_truth_loader import GroundTruth, TestCase, load_ground_truth

DATA_DIR = PROJECT_ROOT / "tests" / "data"
GROUND_TRUTH_PATH = DATA_DIR / "recall_at_k_ground_truth.json"
MAPPING_PATH = DATA_DIR / "id_mapping.json"

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MISSING_BATCH_URL = f"{API_BASE}/api/v1/upload/missing/batch"
FOUND_BATCH_URL = f"{API_BASE}/api/v1/upload/found/batch"

SLEEP_MISSING = float(os.environ.get("UPLOAD_SLEEP_SECONDS", "0.5"))
SLEEP_FOUND = float(os.environ.get("UPLOAD_FOUND_SLEEP_SECONDS", "0.5"))

TIMEOUT = httpx.Timeout(120.0)


def infer_gender(person_name: str) -> str:
    """Deterministically assign gender based on numeric suffix."""
    try:
        idx = int(person_name.split("_")[-1])
        return "male" if idx % 2 else "female"
    except Exception:
        return "male"


def build_location(person_name: str, kind: str) -> str:
    idx_part = "".join(filter(str.isdigit, person_name)) or "0"
    idx = int(idx_part)
    return f"{kind.title()} City #{idx % 10}"


def build_contact(person_name: str, suffix: str) -> str:
    sanitized = person_name.replace(" ", "_").lower()
    return f"{sanitized}@{suffix}"


def compute_photo_years(base_year: int, age_at_disappearance: int, ages: List[int]) -> List[int]:
    photo_years: List[int] = []
    for age in ages:
        if age is None:
            photo_years.append(base_year)
        else:
            delta = age_at_disappearance - age
            photo_year = base_year - delta
            photo_years.append(photo_year)
    return photo_years


def load_image_bytes(image_path: Path) -> bytes:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path.read_bytes()


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class UploadStats:
    missing_success: int = 0
    missing_failed: int = 0
    found_success: int = 0
    found_failed: int = 0


async def upload_missing_case(
    client: httpx.AsyncClient,
    case: TestCase,
    stats: UploadStats,
) -> Optional[str]:
    query_images = case.query_images
    if not query_images:
        print(f"⚠️  Skipping {case.query_id}: no query images.")
        stats.missing_failed += 1
        return None

    # Metadata
    age_at_disappearance = max(case.query_ages) if case.query_ages else 18
    year_disappeared = 2020
    gender = infer_gender(case.query_person_name)
    data = {
        "name": case.query_person_name,
        "age_at_disappearance": str(age_at_disappearance),
        "year_disappeared": str(year_disappeared),
        "gender": gender,
        "location_last_seen": build_location(case.query_person_name, "last_seen"),
        "contact": build_contact(case.query_person_name, "family.test"),
        "additional_info": f"Automated test upload for {case.query_id}",
    }

    # Per-image metadata (photo_year)
    photo_years = compute_photo_years(year_disappeared, age_at_disappearance, case.query_ages)
    image_metadata = [{"photo_year": year} for year in photo_years]
    data["image_metadata_json"] = json.dumps(image_metadata)

    files = []
    for rel_path in query_images:
        image_path = PROJECT_ROOT / rel_path
        image_bytes = load_image_bytes(image_path)
        files.append(
            (
                "images",
                (image_path.name, image_bytes, "image/jpeg"),
            )
        )

    try:
        response = await client.post(
            MISSING_BATCH_URL,
            data=data,
            files=files,
        )
    except httpx.HTTPError as exc:
        print(f"❌ HTTP error uploading {case.query_id}: {exc}")
        stats.missing_failed += 1
        return None

    if response.status_code != 200:
        print(f"❌ Failed to upload {case.query_id}: HTTP {response.status_code} - {response.text}")
        stats.missing_failed += 1
        return None

    payload = response.json()
    if not payload.get("success"):
        print(f"❌ Failed to upload {case.query_id}: {payload.get('message')}")
        stats.missing_failed += 1
        return None

    case_id = payload.get("case_id")
    print(f"✅ Uploaded {case.query_id} → {case_id}")
    stats.missing_success += 1
    return case_id


async def upload_found_person(
    client: httpx.AsyncClient,
    person_name: str,
    image_paths: List[str],
    ages: List[int],
    stats: UploadStats,
) -> Optional[str]:
    if not image_paths:
        print(f"⚠️  Skipping found upload for {person_name}: no images.")
        stats.found_failed += 1
        return None

    current_age_estimate = max(ages) if ages else 25
    gender = infer_gender(person_name)
    current_year = datetime.now().year
    photo_years = [current_year - (current_age_estimate - age) for age in ages]
    image_metadata = [{"photo_year": year} for year in photo_years]

    data = {
        "name": person_name,
        "current_age_estimate": str(current_age_estimate),
        "gender": gender,
        "current_location": build_location(person_name, "current"),
        "finder_contact": build_contact(person_name, "finder.test"),
        "additional_info": "Automated test upload for Recall@K evaluation",
        "image_metadata_json": json.dumps(image_metadata),
    }

    files = []
    for rel_path in image_paths:
        image_path = PROJECT_ROOT / rel_path
        image_bytes = load_image_bytes(image_path)
        files.append(
            (
                "images",
                (image_path.name, image_bytes, "image/jpeg"),
            )
        )

    try:
        response = await client.post(
            FOUND_BATCH_URL,
            data=data,
            files=files,
        )
    except httpx.HTTPError as exc:
        print(f"❌ HTTP error uploading found {person_name}: {exc}")
        stats.found_failed += 1
        return None

    if response.status_code != 200:
        print(f"❌ Failed to upload found {person_name}: HTTP {response.status_code} - {response.text}")
        stats.found_failed += 1
        return None

    payload = response.json()
    if not payload.get("success"):
        print(f"❌ Failed to upload found {person_name}: {payload.get('message')}")
        stats.found_failed += 1
        return None

    found_id = payload.get("case_id") or payload.get("found_id")
    print(f"✅ Uploaded found {person_name} → {found_id}")
    stats.found_success += 1
    return found_id


def gather_found_assets(ground_truth: GroundTruth) -> Dict[str, Dict[str, List]]:
    """Group positive cases by person name."""
    assets: Dict[str, Dict[str, List]] = {}
    for case in ground_truth.test_cases:
        if not case.expected_in_topk:
            continue
        person_name = case.positive_person_name or case.query_person_name
        bucket = assets.setdefault(person_name, {"images": [], "ages": []})
        bucket["images"].extend(case.positive_images)
        bucket["ages"].extend(case.positive_ages or [])
    return assets


async def main():
    ensure_data_dir()
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    stats = UploadStats()

    missing_mapping: Dict[str, Optional[str]] = {}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for case in ground_truth.test_cases:
            case_id = await upload_missing_case(client, case, stats)
            missing_mapping[case.query_id] = case_id
            await asyncio.sleep(SLEEP_MISSING)

        # Upload found persons for positive cases
        found_mapping: Dict[str, Optional[str]] = {}
        found_assets = gather_found_assets(ground_truth)
        for person_name, asset in found_assets.items():
            found_id = await upload_found_person(
                client,
                person_name,
                asset["images"],
                asset["ages"],
                stats,
            )
            found_mapping[person_name] = found_id
            await asyncio.sleep(SLEEP_FOUND)

    output = {
        "missing": missing_mapping,
        "found": found_mapping,
        "meta": {
            "missing_success": stats.missing_success,
            "missing_failed": stats.missing_failed,
            "found_success": stats.found_success,
            "found_failed": stats.found_failed,
            "generated_at": datetime.utcnow().isoformat(),
        },
    }

    MAPPING_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("═══════════════════════════════════════════════")
    print(f"✅ Missing uploads: {stats.missing_success}/{len(ground_truth.test_cases)}")
    print(f"✅ Found uploads: {stats.found_success}/{len(gather_found_assets(ground_truth))}")
    if stats.missing_failed or stats.found_failed:
        print(f"⚠️  Failed uploads → missing: {stats.missing_failed}, found: {stats.found_failed}")
    print(f"💾 Mapping saved to {MAPPING_PATH}")
    print("═══════════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(main())

