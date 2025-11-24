# Child-Specific Face Recognition System

## Overview

This document describes the child-specific face recognition system implemented to address the critical problem where different child faces are often given high similarity scores and incorrectly treated as the same person.

**Problem:** Children's faces are less distinctive than adults, causing false positives in face recognition.

**Solution:** Separate thresholds and evaluation pipelines for children vs adults, with stricter validation for children.

---

## 1. CHILD VALIDATION DATA

### Organizing Child Images

Organize your child images in folders:

```
datasets/children/
  child_001/
    img1.jpg
    img2.jpg
    ...
  child_002/
    img1.jpg
    ...
```

Or use multiple root folders:
```
datasets/children/
datasets/child_val/
```

**Age Information (Optional):**
- Age can be extracted from folder names (e.g., `child_001_age_5`, `age_7_child_002`)
- Or from filenames (e.g., `img_age_5.jpg`)
- Supported patterns: `age_5`, `5_years`, `5yo`, `child_5`
- If age is not found, it will be left empty in the CSV

### Building Validation Pairs

Run the child-specific validation pair builder:

```bash
python tools/build_validation_pairs_children.py
```

**With custom options:**
```bash
python tools/build_validation_pairs_children.py \
  --roots datasets/children datasets/child_val \
  --max-same 50 \
  --max-diff 1000 \
  --output datasets/validation_pairs_children.csv
```

**Output:**
- CSV file: `datasets/validation_pairs_children.csv`
- Format: `person_id_1,image_path_1,age_1,person_id_2,image_path_2,age_2,label`
- `label=1` for same child, `label=0` for different children

**What it does:**
- Scans child folders
- Generates same-child pairs (within each child folder)
- Generates different-child pairs (cross-child, balanced)
- Logs counts and saves CSV

---

## 2. CHILD THRESHOLD EVALUATION

### Running Child-Only Threshold Sweep

Run the child-specific threshold evaluation:

```bash
python tests/eval_threshold_sweep_children.py
```

**With custom options:**
```bash
python tests/eval_threshold_sweep_children.py \
  --csv datasets/validation_pairs_children.csv \
  --min-threshold 0.30 \
  --max-threshold 0.80 \
  --step 0.01 \
  --target-fpr 0.01 \
  --output tests/threshold_sweep_children_results.csv
```

### Interpreting Results

**Metrics to look at:**

1. **TPR (True Positive Rate)**: Recall on same-child pairs
   - Higher is better
   - Measures how many same-child pairs are correctly identified

2. **FPR (False Positive Rate)**: Fraction of different-children pairs wrongly accepted
   - Lower is better
   - Measures false positives (different children treated as same)

3. **F1-score**: Harmonic mean of Precision and TPR
   - Balanced metric
   - Higher is better

4. **EER (Equal Error Rate)**: Threshold where FPR ≈ FNR
   - Lower EER is better
   - Balance point between false positives and false negatives

**Output includes:**
- Console table with all metrics for each threshold
- EER threshold and value
- Threshold that maximizes F1-score
- Threshold with FPR ≤ target (e.g., 0.01)
- **Recommended child threshold for search** (most important!)

### Recommended Threshold Computation

The script computes `recommended_child_threshold_for_search` using this logic:

1. **Primary strategy**: Among thresholds with FPR ≤ target_child_FPR (default 0.01):
   - Pick the one with maximum F1-score
   - This balances recall (TPR) and precision

2. **Fallback 1**: If no threshold satisfies FPR ≤ target:
   - Try with relaxed FPR (2x target, e.g., 0.02)
   - Pick max F1 among those

3. **Fallback 2**: If still no threshold:
   - Use threshold with maximum F1 overall
   - This ensures we always have a recommendation

**Example output:**
```
RECOMMENDED CHILD THRESHOLD FOR MISSING-PERSON SEARCH
====================================================

Threshold: 0.450
Reason: FPR <= 0.01 with max F1

Metrics at this threshold:
  TPR (same-child recall): 0.850
  FPR (different-children false accept): 0.008
  Precision: 0.920
  F1-score: 0.884

>>> Use this threshold value for face_search_threshold_child in Settings <<<
```

**Use this threshold value** in your `.env` file or Settings configuration.

---

## 3. CONFIG & SEARCH INTEGRATION

### Settings Configuration

The system now has separate thresholds for adults and children:

**In `api/config.py` (Settings class):**

```python
face_search_threshold_adult: float = Field(
    default=0.30,
    description="Face similarity threshold for ADULT missing-person search"
)

face_search_threshold_child: float = Field(
    default=0.45,
    description="Face similarity threshold for CHILD missing-person search"
)
```

**Environment Variables:**
- `FACE_SEARCH_THRESHOLD_ADULT=0.30` (or your adult threshold from eval_threshold_sweep.py)
- `FACE_SEARCH_THRESHOLD_CHILD=0.45` (or your child threshold from eval_threshold_sweep_children.py)

**Example `.env` file:**
```env
FACE_SEARCH_THRESHOLD_ADULT=0.300
FACE_SEARCH_THRESHOLD_CHILD=0.450
```

### Where Thresholds Are Applied

**In `services/bilateral_search.py`:**

1. **Age Classification:**
   - `_is_child(metadata)` function determines if a person is a child (age < 18)
   - Checks: `age_at_disappearance`, `current_age_estimate`, `estimated_current_age`, `age`
   - If age is unknown, defaults to **adult** (more lenient)

2. **Threshold Selection:**
   - `_get_face_threshold_for_person(metadata)` returns appropriate threshold
   - Children → `face_threshold_child`
   - Adults (or unknown) → `face_threshold_adult`

3. **Search Methods:**
   - `search_for_missing()`: Uses age-appropriate threshold for each candidate
   - `search_for_found()`: Uses age-appropriate threshold for each candidate
   - Applied in filtering step: `if m['face_similarity'] >= threshold_for_match`

**Code flow:**
```python
# For each candidate match:
match_metadata = m.get('payload', {})
threshold_for_match = self._get_face_threshold_for_person(match_metadata)

if m['face_similarity'] >= threshold_for_match:
    # Include in results
```

### Human-in-the-Loop Pattern

**The system enforces a strict TOP-K + HUMAN-IN-THE-LOOP pattern:**

1. **No hard yes/no decisions:**
   - The system does NOT return a boolean "same person = true/false"
   - Instead, returns a ranked list of candidates with scores

2. **Search flow:**
   ```
   Query embedding
     ↓
   Qdrant search (top_k_raw = 50, low score_threshold)
     ↓
   Apply age-appropriate face_search_threshold_* (filter)
     ↓
   Sort by similarity
     ↓
   Return top_k_to_return (e.g., 10-20) to API client
   ```

3. **API Response:**
   - Returns list of candidates with:
     - `id`: Candidate ID
     - `face_similarity`: Similarity score
     - `metadata_similarity`: Metadata match score
     - `combined_score`: Overall score
     - `confidence_score`: Confidence level
     - `metadata`: Full metadata (age, location, etc.)
   - **Human operator reviews and decides**

4. **For Children:**
   - Stricter threshold (higher) reduces false positives
   - Still returns ranked candidates (not a single yes/no)
   - Human must verify all matches

### Validation Logic

**Additional strict validation for children** (in `_validate_match()`):

- **Very high similarity (>98%)**: Requires >75% metadata support
- **High similarity (>95%)**: Requires >70% metadata support
- **Moderate-high similarity (>90%)**: Requires >60% metadata support

This prevents false positives even when face similarity is very high.

---

## Summary

### Key Points

1. **Separate evaluation**: Children require their own threshold sweep
2. **Stricter thresholds**: Children use higher thresholds (typically 0.40-0.55 vs 0.25-0.35 for adults)
3. **Age-based selection**: System automatically uses correct threshold based on person's age
4. **Human-in-the-loop**: Always returns ranked candidates, never auto-decides
5. **Metadata validation**: Children require stronger metadata support for high similarity matches

### Workflow

1. **Prepare child validation data:**
   ```bash
   python tools/build_validation_pairs_children.py
   ```

2. **Run child threshold evaluation:**
   ```bash
   python tests/eval_threshold_sweep_children.py
   ```

3. **Get recommended threshold** from output

4. **Update `.env` file:**
   ```env
   FACE_SEARCH_THRESHOLD_CHILD=0.450  # Use recommended value
   ```

5. **Restart API server** to apply new threshold

6. **Test with child images** to verify false positives are reduced

---

## Files Created/Modified

### New Files
- `tools/build_validation_pairs_children.py` - Child validation pair builder
- `tests/eval_threshold_sweep_children.py` - Child-only threshold evaluation
- `CHILD_SPECIFIC_SYSTEM.md` - This documentation

### Modified Files
- `api/config.py` - Added `face_search_threshold_adult` and `face_search_threshold_child`
- `services/bilateral_search.py` - Added age-based threshold selection
- `api/dependencies.py` - Updated to pass separate thresholds

---

## Troubleshooting

**Q: How do I know if a person is classified as child or adult?**
- Check the logs: `_is_child()` function logs age detection
- Or check metadata: age < 18 = child, age >= 18 or unknown = adult

**Q: What if age is not in metadata?**
- System defaults to **adult** (more lenient threshold)
- This is conservative: unknown age = use adult threshold

**Q: Can I change the age cutoff (currently 18)?**
- Yes, modify `_is_child()` function in `bilateral_search.py`
- Change `return age < 18` to your desired cutoff

**Q: The recommended threshold seems too high/low.**
- Adjust `--target-fpr` in eval script (default 0.01)
- Lower target FPR = higher threshold (stricter)
- Higher target FPR = lower threshold (more lenient)

