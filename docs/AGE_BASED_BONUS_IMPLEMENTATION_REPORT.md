# Age-Based Threshold/Bonus Implementation Report

**Date:** November 27, 2025  
**System:** Missing Person AI - Multi-Image Matching  
**Investigation Scope:** Age bracket bonus, NULL age handling, data flow, configuration

---

## Executive Summary

### Is Age-Based Bonus Implemented?
**✅ YES** - Age-based bonus is fully implemented and working.

### Current Implementation Quality
**✅ GOOD** - Implementation is production-ready with:
- Configurable age brackets (3 tiers)
- Proper NULL handling (skips pairs with missing ages)
- Automatic age calculation with fallback
- Logging for debugging

### Key Findings
- ✅ **Age bracket bonus**: Fully implemented with 3 tiers (0-3y, 4-7y, 8-15y, 16+y)
- ⚠️ **NULL handling**: Images with missing `age_at_photo` are **SKIPPED** from matching (not used)
- ✅ **Age calculation**: Automatic fallback to `age_at_disappearance` when `photo_year` not provided
- ⚠️ **Configuration**: `age_bracket_bonus` is **hardcoded** (default 0.02), not configurable via Settings

---

## Implementation Details

### 1. Location

**File:** `services/multi_image_aggregation.py`  
**Function:** `_apply_age_bracket_bonus()`  
**Line:** 397-430

**Service Initialization:** `services/multi_image_aggregation.py:100-137`  
**Usage:** `services/multi_image_aggregation.py:304-307`

---

### 2. Logic

#### Age Brackets

The bonus is applied based on age gap between query and target images:

```python
if age_gap <= 3:
    bonus = self.age_bracket_bonus  # Full bonus (default: 0.020 = +2%)
elif age_gap <= 7:
    bonus = self.age_bracket_bonus * 0.7  # 70% bonus (default: 0.014 = +1.4%)
elif age_gap <= 15:
    bonus = self.age_bracket_bonus * 0.4  # 40% bonus (default: 0.008 = +0.8%)
else:
    bonus = 0.0  # No bonus for large gaps (>15 years)
```

**Default Bonus Values:**
- **Tier 1 (0-3 years):** +0.020 (+2.0%)
- **Tier 2 (4-7 years):** +0.014 (+1.4%)
- **Tier 3 (8-15 years):** +0.008 (+0.8%)
- **Tier 4 (16+ years):** +0.000 (no bonus)

#### Application

```python
adjusted_similarity = min(1.0, similarity + bonus)
```

The bonus is **added** to the raw similarity score, capped at 1.0.

**Example:**
- Raw similarity: 0.577
- Age gap: 2 years → Tier 1 bonus: +0.020
- Adjusted similarity: min(1.0, 0.577 + 0.020) = **0.597**

---

### 3. NULL Handling

#### Current Behavior When `age_at_photo` is None

**✅ SKIP IMAGE PAIR** - Images with missing `age_at_photo` are **excluded** from matching.

**Code Location:** `services/multi_image_aggregation.py:275-293`

```python
# Skip if age is missing
if query_age is None:
    logger.warning(f"Query image {query_image_id} missing age_at_photo, skipping")
    continue

# Skip if age is missing
if target_age is None:
    logger.warning(f"Target image {target_image_id} missing age_at_photo, skipping")
    continue
```

**Impact:**
- If query image has `age_at_photo=None`, that image is **not used** for matching
- If target image has `age_at_photo=None`, that image is **not used** for matching
- Other images with valid ages are still used
- If **all** images have `None` ages, no matching occurs (returns zero-score result)

**Rationale:**
- Age bracket bonus requires age information
- Better to skip than apply incorrect bonus
- Prevents errors from None arithmetic

#### Test Scenarios

| Scenario | Query Age | Target Age | Behavior |
|----------|-----------|------------|----------|
| 1. Both provided | 25 | 27 | ✅ Used, bonus applied (gap=2y → +0.020) |
| 2. Query missing | None | 27 | ⚠️ Query image **skipped** |
| 3. Target missing | 25 | None | ⚠️ Target image **skipped** |
| 4. Both missing | None | None | ⚠️ Image pair **skipped** |

**Note:** In scenarios 2-4, other image pairs with valid ages are still processed.

---

### 4. Data Flow

#### Step 1: Upload

**Location:** `api/routes/upload.py:642-652`

```python
# Calculate age at photo (ALWAYS, even if no face detected)
try:
    age_at_photo = calculate_age_at_photo(
        photo_year=img_meta.get('photo_year') if img_meta else None,
        year_disappeared=shared_params['year_disappeared'],
        age_at_disappearance=shared_params['age_at_disappearance']
    )
except Exception as e:
    logger.error(f"Age calculation failed for image {idx}: {e}")
    # Fallback to age_at_disappearance
    age_at_photo = shared_params['age_at_disappearance']
```

**Age Calculation Logic** (`utils/image_helpers.py:21-120`):
- If `photo_year` provided: `age_at_photo = age_at_disappearance - (year_disappeared - photo_year)`
- If `photo_year` is None: `age_at_photo = age_at_disappearance` (assume recent photo)
- Validates photo_year (not in future, not after disappearance)
- Clamps result to 0-120

**Result:** `age_at_photo` is **always** calculated (never None) during upload.

#### Step 2: Storage

**Location:** `api/routes/upload.py:920-945`

```python
payload = {
    # Per-image metadata
    "age_at_photo": img_result['age_at_photo'],  # Always present (calculated)
    "photo_year": img_result.get('photo_year'),  # Optional (may be None)
    
    # Shared person metadata
    "age_at_disappearance": age_at_disappearance,
    "year_disappeared": year_disappeared,
    # ... other fields ...
}
```

**Qdrant Payload Structure:**
```json
{
  "age_at_photo": 15,  // Always integer (0-120)
  "photo_year": 2010,  // Optional (may be None)
  "age_at_disappearance": 25,
  "year_disappeared": 2020,
  // ... other fields ...
}
```

**Note:** `age_at_photo` is **always stored** (never None) because it's calculated during upload.

#### Step 3: Retrieval

**Location:** `services/bilateral_search.py:866-871`

```python
candidate_images.append({
    'image_id': point['payload'].get('image_id', point['id']),
    'embedding': np.array(point['vector']),
    'age_at_photo': point['payload'].get('age_at_photo', 0),  # ← Default to 0 if missing
    'case_id': found_id
})
```

**⚠️ ISSUE FOUND:** Default value is `0` if `age_at_photo` is missing from payload.

**Impact:**
- If Qdrant payload somehow has missing `age_at_photo`, it defaults to `0`
- This would cause incorrect age gap calculations (e.g., query_age=25, target_age=0 → gap=25y)
- However, this should never happen because `age_at_photo` is always calculated during upload

#### Step 4: Aggregation

**Location:** `services/multi_image_aggregation.py:265-307`

```python
for query_img in query_images:
    query_age = query_img.get('age_at_photo')
    
    # Skip if age is missing
    if query_age is None:
        logger.warning(f"Query image {query_image_id} missing age_at_photo, skipping")
        continue
    
    for target_img in target_images:
        target_age = target_img.get('age_at_photo')
        
        # Skip if age is missing
        if target_age is None:
            logger.warning(f"Target image {target_image_id} missing age_at_photo, skipping")
            continue
        
        # Compute similarity
        similarity = self._cosine_similarity(query_embedding, target_embedding)
        
        # Apply age-bracket preference bonus if enabled
        if self.age_bracket_preference_enabled:
            age_gap = abs(query_age - target_age)
            similarity = self._apply_age_bracket_bonus(similarity, age_gap)
```

#### Step 5: Bonus Calculation

**Location:** `services/multi_image_aggregation.py:397-430`

```python
def _apply_age_bracket_bonus(self, similarity: float, age_gap: int) -> float:
    # Define age brackets with bonuses
    if age_gap <= 3:
        bonus = self.age_bracket_bonus  # Full bonus
    elif age_gap <= 7:
        bonus = self.age_bracket_bonus * 0.7  # 70% bonus
    elif age_gap <= 15:
        bonus = self.age_bracket_bonus * 0.4  # 40% bonus
    else:
        bonus = 0.0  # No bonus
    
    # Apply bonus (capped at 1.0)
    adjusted_similarity = min(1.0, similarity + bonus)
    
    return adjusted_similarity
```

---

### 5. Configuration

#### Current Configuration Status

**⚠️ HARDCODED** - `age_bracket_bonus` is **not** configurable via Settings.

**Default Values:**
- `age_bracket_bonus = 0.02` (hardcoded in `MultiImageAggregationService.__init__()`)
- `age_bracket_preference_enabled = True` (hardcoded)

**Location:** `services/multi_image_aggregation.py:100-130`

```python
def __init__(
    self,
    consistency_bonus_weight: float = 0.05,
    good_match_threshold: float = 0.25,
    age_bracket_preference_enabled: bool = True,  # ← Hardcoded default
    age_bracket_bonus: float = 0.02  # ← Hardcoded default
) -> None:
```

**Factory Function:** `services/multi_image_aggregation.py:582-602`

```python
def get_aggregation_service(
    consistency_bonus_weight: float = 0.05,
    good_match_threshold: float = 0.25
    # ⚠️ age_bracket_bonus NOT configurable here!
) -> MultiImageAggregationService:
    global _default_service
    if _default_service is None:
        _default_service = MultiImageAggregationService(
            consistency_bonus_weight=consistency_bonus_weight,
            good_match_threshold=good_match_threshold
            # Uses default age_bracket_bonus=0.02
        )
    return _default_service
```

**Settings File:** `api/config.py`
- ❌ No `age_bracket_bonus` field
- ❌ No `age_bracket_preference_enabled` field

**Environment Variables:**
- ❌ No environment variable support

---

## Issues Found

### Issue 1: NULL Age Handling - Skip Behavior

**Severity:** ⚠️ **MEDIUM**

**Description:**
When `age_at_photo` is None, the image pair is **skipped** entirely from matching. This means:
- Images without age metadata are **not used** for matching
- This could reduce matching accuracy if age calculation fails
- However, since `age_at_photo` is **always calculated** during upload, this should rarely occur

**Current Behavior:**
```python
if query_age is None:
    logger.warning(f"Query image {query_image_id} missing age_at_photo, skipping")
    continue  # ← Image pair skipped
```

**Problem:**
- If age calculation fails during upload, fallback uses `age_at_disappearance`
- But if somehow `age_at_photo` becomes None later (data corruption, migration), images are skipped
- This is **defensive** but may hide data quality issues

**Recommendation:**
- ✅ Current behavior is **acceptable** (defensive, prevents errors)
- Consider adding metrics to track skipped images
- Consider fallback to `age_at_disappearance` if `age_at_photo` is None (but this requires person-level metadata)

---

### Issue 2: Default Value in Retrieval

**Severity:** ⚠️ **LOW**

**Description:**
In `bilateral_search.py:869`, if `age_at_photo` is missing from Qdrant payload, it defaults to `0`:

```python
'age_at_photo': point['payload'].get('age_at_photo', 0)  # ← Defaults to 0
```

**Problem:**
- Defaulting to `0` could cause incorrect age gap calculations
- However, this should never happen because `age_at_photo` is always calculated during upload

**Recommendation:**
- Change default to `None` to trigger skip logic:
  ```python
  'age_at_photo': point['payload'].get('age_at_photo')  # None if missing
  ```
- This ensures NULL handling logic is triggered if data is corrupted

---

### Issue 3: Configuration Not Exposed

**Severity:** ⚠️ **LOW**

**Description:**
`age_bracket_bonus` is hardcoded and not configurable via Settings or environment variables.

**Problem:**
- Cannot adjust bonus values without code changes
- Cannot disable age bonus via configuration
- Harder to tune for different use cases

**Recommendation:**
- Add to `api/config.py`:
  ```python
  age_bracket_bonus: float = Field(
      default=0.02,
      ge=0.0,
      le=0.1,
      description="Age bracket bonus for similar-age matches"
  )
  age_bracket_preference_enabled: bool = Field(
      default=True,
      description="Enable age bracket preference scoring"
  )
  ```
- Update `get_aggregation_service()` to accept these from Settings

---

## Recommendations

### Priority 1: [MEDIUM] Fix Default Value in Retrieval

**Action:** Change default `age_at_photo` from `0` to `None` in retrieval

**File to modify:** `services/bilateral_search.py:869`

**Change:**
```python
# OLD:
'age_at_photo': point['payload'].get('age_at_photo', 0)

# NEW:
'age_at_photo': point['payload'].get('age_at_photo')  # None if missing
```

**Estimated effort:** 5 minutes

**Impact:** Ensures NULL handling logic is triggered if data is corrupted

---

### Priority 2: [LOW] Add Configuration Support

**Action:** Add `age_bracket_bonus` and `age_bracket_preference_enabled` to Settings

**Files to modify:**
1. `api/config.py` - Add fields
2. `services/multi_image_aggregation.py` - Update `get_aggregation_service()` to accept Settings

**Change:**
```python
# api/config.py
age_bracket_bonus: float = Field(
    default=0.02,
    ge=0.0,
    le=0.1,
    description="Age bracket bonus for similar-age matches (0-0.1)"
)
age_bracket_preference_enabled: bool = Field(
    default=True,
    description="Enable age bracket preference scoring"
)
```

**Estimated effort:** 30 minutes

**Impact:** Allows runtime configuration without code changes

---

### Priority 3: [LOW] Add Metrics for Skipped Images

**Action:** Track how many image pairs are skipped due to missing ages

**File to modify:** `services/multi_image_aggregation.py`

**Change:**
```python
skipped_query_count = 0
skipped_target_count = 0

# ... in loop ...
if query_age is None:
    skipped_query_count += 1
    logger.warning(...)
    continue

# ... at end ...
if skipped_query_count > 0 or skipped_target_count > 0:
    logger.warning(
        f"Skipped {skipped_query_count} query images and "
        f"{skipped_target_count} target images due to missing age_at_photo"
    )
```

**Estimated effort:** 15 minutes

**Impact:** Better visibility into data quality issues

---

## Code References

### Age Bracket Bonus Implementation

1. **`services/multi_image_aggregation.py:397-430`** - `_apply_age_bracket_bonus()`
   - Main bonus calculation logic
   - 3-tier bonus system

2. **`services/multi_image_aggregation.py:304-307`** - Bonus application
   - Called during pairwise similarity computation
   - Only applied if `age_bracket_preference_enabled=True`

3. **`services/multi_image_aggregation.py:100-137`** - Service initialization
   - Default values: `age_bracket_bonus=0.02`, `age_bracket_preference_enabled=True`
   - Validation: `age_bracket_bonus` must be in [0, 0.1]

### NULL Handling

4. **`services/multi_image_aggregation.py:275-293`** - NULL age checks
   - Skips query images with `age_at_photo=None`
   - Skips target images with `age_at_photo=None`
   - Logs warnings for debugging

### Age Calculation

5. **`utils/image_helpers.py:21-120`** - `calculate_age_at_photo()`
   - Calculates age from `photo_year`, `year_disappeared`, `age_at_disappearance`
   - Fallback to `age_at_disappearance` if `photo_year` is None
   - Validates inputs and clamps result to 0-120

6. **`api/routes/upload.py:642-652`** - Age calculation during upload
   - Always calculates `age_at_photo` (never None)
   - Fallback to `age_at_disappearance` on error

### Storage & Retrieval

7. **`api/routes/upload.py:930`** - Storage in Qdrant payload
   - `age_at_photo` always stored (calculated during upload)

8. **`services/bilateral_search.py:869`** - Retrieval from Qdrant
   - Extracts `age_at_photo` from payload
   - ⚠️ Defaults to `0` if missing (should be `None`)

### Service Factory

9. **`services/multi_image_aggregation.py:582-602`** - `get_aggregation_service()`
   - Singleton factory function
   - ⚠️ Does not accept `age_bracket_bonus` parameter (uses default 0.02)

---

## Test Evidence

From test logs, age bracket bonus is working correctly:

```
Age bracket bonus applied: age_gap=2y, bonus=0.020, sim=0.577 -> 0.597
Age bracket bonus applied: age_gap=11y, bonus=0.008, sim=0.482 -> 0.490
```

**Analysis:**
- Age gap 2y → Tier 1 bonus (+0.020) ✅
- Age gap 11y → Tier 3 bonus (+0.008) ✅
- Bonuses applied correctly ✅

---

## Conclusion

The age-based bonus implementation is **production-ready** and working correctly. The system:

- ✅ Applies bonuses based on age gaps (3 tiers)
- ✅ Handles NULL ages defensively (skips pairs)
- ✅ Always calculates `age_at_photo` during upload (fallback logic)
- ✅ Logs bonus applications for debugging

**Minor improvements** (configuration, default values) would enhance flexibility but are not critical.

**Overall Assessment:** ✅ **GOOD** - Ready for production use.

---

**Report Generated:** November 27, 2025  
**Investigator:** AI Assistant  
**Codebase Version:** Phase 3 (Multi-Image Feature Complete)

