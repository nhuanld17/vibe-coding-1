# Top-K Ranking Implementation Report

**Date:** November 27, 2025  
**System:** Missing Person AI - Multi-Image Matching  
**Investigation Scope:** Top-K ranking, sorting, filtering, and response format

---

## Executive Summary

### Is Top-K Ranking Implemented?
**✅ YES** - Top-K ranking is fully implemented across all search endpoints.

### Current Implementation Quality
**✅ GOOD** - Implementation is production-ready with:
- Configurable K parameter (1-100)
- Proper sorting by combined similarity score
- Threshold filtering before ranking
- Multi-image aggregation support
- Detailed response format with confidence scores

### Key Findings
- ✅ **Top-K ranking**: Fully implemented with configurable limit (default: 10, max: 100)
- ✅ **Sorting**: Results sorted by `combined_score` (descending) - face similarity + metadata similarity
- ✅ **Filtering**: Multi-stage filtering (threshold → validation → top-K)
- ⚠️ **Pagination**: Not implemented (returns all top-K in single response)
- ✅ **Multi-image support**: Aggregates multiple images per person before ranking

---

## Detailed Findings

### 1. API Endpoint Analysis

#### 1.1 Search Endpoints

**Endpoint 1: GET `/api/v1/search/missing/{case_id}`**
- **Location:** `api/routes/search.py:147`
- **Top-K Support:** ✅ YES
- **Default K Value:** `1` (query parameter)
- **Max K Value:** `100` (enforced by FastAPI validation)
- **Request Format:**
  ```python
  GET /api/v1/search/missing/{case_id}?limit=10
  ```
- **Response Format:**
  ```json
  {
    "success": true,
    "message": "Found missing person with case_id '...' and 5 potential match(es)",
    "missing_person": {...},
    "matches": [
      {
        "id": "FOUND_001",
        "face_similarity": 0.85,
        "metadata_similarity": 0.78,
        "combined_score": 0.82,
        "confidence_level": "HIGH",
        "confidence_score": 0.82,
        "explanation": {...},
        "contact": "...",
        "metadata": {...},
        "image_url": "..."
      },
      ...
    ],
    "total_found": 5,
    "search_parameters": {
      "limit": 10,
      "threshold": 0.45
    },
    "processing_time_ms": 123.45
  }
  ```

**Endpoint 2: GET `/api/v1/search/found/{found_id}`**
- **Location:** `api/routes/search.py:276`
- **Top-K Support:** ✅ YES
- **Default K Value:** `1`
- **Max K Value:** `100`
- **Request/Response:** Same format as above (symmetric)

**Endpoint 3: POST `/api/v1/upload/missing/batch`**
- **Location:** `api/routes/upload.py:590`
- **Top-K Support:** ✅ YES (via `settings.top_k_matches`)
- **Default K Value:** `10` (from config)
- **Response:** Includes `potential_matches: List[MatchResult]` in `MultiImageUploadResponse`

**Endpoint 4: POST `/api/v1/upload/found/batch`**
- **Location:** `api/routes/upload.py:590` (symmetric)
- **Top-K Support:** ✅ YES
- **Default K Value:** `10`

#### 1.2 User Customization

**Can user customize K?**
- ✅ **YES** for search endpoints: via `?limit=K` query parameter
- ⚠️ **NO** for upload endpoints: uses `settings.top_k_matches` (not user-configurable)

**Recommendation:**
- Add optional `limit` parameter to batch upload endpoints for user customization

---

### 2. Ranking Logic

#### 2.1 Sorting Method

**Primary Sort Key:** `combined_score` (descending)

**Location:** `services/bilateral_search.py:924`
```python
aggregated_results.sort(key=lambda x: x['combined_score'], reverse=True)
```

**Combined Score Calculation:**
```python
combined_score = (face_weight * face_similarity + 
                  metadata_weight * metadata_similarity)
```
- `face_weight = 0.7` (default)
- `metadata_weight = 0.3` (default)

**For Multi-Image Matches:**
- Face similarity = aggregated from multiple images (best + mean + consistency)
- Location: `services/multi_image_aggregation.py:476-516`

#### 2.2 Score Calculation

**Single-Image Search:**
1. **Face Similarity:** Cosine similarity between embeddings (0-1)
2. **Metadata Similarity:** Weighted match on age, gender, location (0-1)
3. **Combined Score:** `0.7 * face + 0.3 * metadata`

**Multi-Image Search:**
1. **Aggregated Face Similarity:**
   - Best pairwise similarity: `max(all_pair_scores)`
   - Mean similarity: `mean(top_k_pair_scores)`
   - Consistency score: `num_good_matches / total_pairs`
   - Final face score: `0.7 * best + 0.2 * mean + 0.1 * consistency`
2. **Metadata Similarity:** Same as single-image
3. **Combined Score:** `0.7 * aggregated_face + 0.3 * metadata`

#### 2.3 Threshold Filtering

**Multi-Stage Filtering:**

**Stage 1: Initial Vector Search**
- **Location:** `services/bilateral_search.py:103-109`
- **Threshold:** `initial_search_threshold = 0.40` (configurable)
- **Purpose:** Filter low-similarity candidates at database level
- **Method:** Qdrant vector search with `score_threshold`

**Stage 2: Validation**
- **Location:** `services/bilateral_search.py:126`
- **Method:** `_validate_match()` - rejects suspicious false positives
- **Checks:**
  - Age consistency
  - Gender match
  - Location plausibility
  - Face similarity vs metadata similarity ratio

**Stage 3: Threshold Filtering**
- **Location:** `services/bilateral_search.py:131-134`
- **Conditions (OR logic):**
  - `face_similarity >= age_appropriate_threshold` OR
  - `combined_score >= 0.55` OR
  - `face_similarity >= 0.50 AND metadata_similarity >= 0.60`

**Stage 4: Top-K Limit**
- **Location:** `services/bilateral_search.py:140`
- **Method:** `filtered_matches[:limit]`
- **Applied:** After all filtering, returns top K results

#### 2.4 What Happens if < K Candidates Pass Threshold?

**Behavior:** Returns fewer than K results (no padding)

**Example:**
- Request: `limit=10`
- Candidates passing threshold: 3
- Response: Returns 3 matches (not padded to 10)

**Rationale:** Better to return fewer high-quality matches than pad with low-quality ones.

---

### 3. Database Query

#### 3.1 Search Strategy

**Does it retrieve ALL candidates first, then rank?**
- ⚠️ **PARTIALLY** - Uses two-stage approach:

**Stage 1: Vector Search (Database-Level)**
- **Location:** `services/vector_db.py:search_similar_faces()`
- **Method:** Qdrant vector similarity search
- **Limit:** `limit * 5` (for single-image) or `limit * 10` (for multi-image)
- **Purpose:** Pre-filter at database level using vector similarity
- **Performance:** Efficient (uses Qdrant's optimized vector search)

**Stage 2: In-Memory Reranking**
- **Location:** `services/bilateral_search.py:114-116`
- **Method:** `_rerank_with_metadata()`
- **Purpose:** Combine face similarity with metadata matching
- **Performance:** Fast (operates on small candidate set)

#### 3.2 Performance Considerations

**For Large Database (10K+ persons):**
- ✅ **Efficient:** Vector search uses Qdrant's HNSW index (logarithmic complexity)
- ✅ **Scalable:** Initial search limit prevents loading all candidates
- ⚠️ **Bottleneck:** Metadata reranking (O(n) where n = initial_limit * 5)

**Optimization Opportunities:**
1. **Database-level metadata filtering:** Use Qdrant filters for age/gender
2. **Caching:** Cache metadata similarity calculations
3. **Parallel processing:** Process multiple candidates in parallel

---

### 4. Current Implementation Status

| Feature | Status | Details |
|---------|--------|---------|
| **Top-K ranking implemented** | ✅ | Fully implemented with configurable limit |
| **Results sorted by similarity** | ✅ | Sorted by `combined_score` (descending) |
| **Configurable K parameter** | ✅ | Query parameter `limit` (1-100) |
| **Pagination support** | ❌ | Not implemented (returns all top-K) |
| **Threshold filtering (pre-ranking)** | ✅ | Multi-stage filtering before ranking |
| **Detailed candidate info in response** | ✅ | Full `MatchResult` with confidence scores |
| **Multi-image aggregation** | ✅ | Aggregates multiple images per person |
| **Rank field in response** | ❌ | Not explicitly included (can infer from array index) |

---

## Code Locations

### Main Search Endpoints
- **Single-image search:** `api/routes/search.py:147-273` (missing), `276-397` (found)
- **Multi-image search:** `api/routes/upload.py:590-800` (batch upload endpoints)

### Ranking Logic
- **Bilateral search:** `services/bilateral_search.py:72-147` (single-image)
- **Multi-image search:** `services/bilateral_search.py:790-933` (multi-image)
- **Sorting:** `services/bilateral_search.py:924` (line 924)

### Aggregation Service
- **Multi-image aggregation:** `services/multi_image_aggregation.py:139-245`
- **Final score calculation:** `services/multi_image_aggregation.py:476-516`

### Response Builder
- **Format match results:** `api/routes/upload.py:87-140` (`format_match_results()`)
- **Response models:** `api/schemas/models.py:228-264`

### Configuration
- **Default K value:** `api/config.py:65-70` (`top_k_matches = 10`)
- **Thresholds:** `api/config.py:75-95` (various thresholds)

---

## Recommendations

### What Works Well

1. **Multi-Stage Filtering**
   - Efficient database-level pre-filtering
   - In-memory reranking for accuracy
   - Validation prevents false positives

2. **Multi-Image Aggregation**
   - Handles multiple images per person correctly
   - Aggregates scores intelligently (best + mean + consistency)
   - Age-bracket preference improves accuracy

3. **Flexible K Parameter**
   - User-configurable via query parameter
   - Reasonable defaults (10 for uploads, 1 for searches)
   - Validation prevents abuse (max 100)

4. **Detailed Response Format**
   - Includes confidence scores and explanations
   - Provides metadata for human verification
   - Clear structure for frontend integration

### Missing/Needs Improvement

1. **Pagination Support**
   - **Issue:** Returns all top-K in single response
   - **Impact:** May be slow for large K values
   - **Priority:** MEDIUM

2. **Rank Field Missing**
   - **Issue:** No explicit `rank` field in response
   - **Impact:** Frontend must infer rank from array index
   - **Priority:** LOW

3. **Upload Endpoint K Customization**
   - **Issue:** Batch upload uses fixed `settings.top_k_matches`
   - **Impact:** Users cannot customize K for upload searches
   - **Priority:** LOW

4. **Database-Level Metadata Filtering**
   - **Issue:** Metadata filtering done in-memory after vector search
   - **Impact:** May load unnecessary candidates from database
   - **Priority:** MEDIUM (for large databases)

### Suggested Improvements (Priority Order)

#### 1. [HIGH] Add Rank Field to Response
**File:** `api/schemas/models.py`
**Change:**
```python
class MatchResult(BaseModel):
    rank: int = Field(..., description="Rank of this match (1-based)", ge=1)
    # ... existing fields ...
```

**Update:** `api/routes/upload.py:format_match_results()` to include rank

**Impact:** Improves frontend UX, clearer API contract

---

#### 2. [MEDIUM] Add Pagination Support
**Files:** `api/routes/search.py`, `api/schemas/models.py`
**Changes:**
- Add `page` and `page_size` query parameters
- Update `SearchResponse` to include pagination metadata:
  ```python
  class SearchResponse(BaseModel):
      # ... existing fields ...
      pagination: Optional[PaginationInfo] = Field(None)
  
  class PaginationInfo(BaseModel):
      page: int
      page_size: int
      total_pages: int
      total_results: int
  ```

**Impact:** Better performance for large result sets

---

#### 3. [MEDIUM] Database-Level Metadata Filtering
**File:** `services/vector_db.py`
**Change:** Use Qdrant filters for age/gender before vector search
```python
filters = {
    "age_at_disappearance": {"$gte": min_age, "$lte": max_age},
    "gender": gender
}
vector_matches = self.client.search(
    ...,
    query_filter=Filter(**filters)
)
```

**Impact:** Reduces database load, faster queries

---

#### 4. [LOW] Add K Parameter to Upload Endpoints
**File:** `api/routes/upload.py`
**Change:** Add optional `limit` parameter to batch upload endpoints
```python
async def upload_missing_person_batch(
    ...,
    limit: int = Form(default=None, description="Max matches to return")
):
    if limit is None:
        limit = settings.top_k_matches
    # ... use limit in search ...
```

**Impact:** More flexibility for users

---

#### 5. [LOW] Add Total Candidates Count
**File:** `api/schemas/models.py`
**Change:** Add `total_candidates_considered` to `SearchResponse`
```python
class SearchResponse(BaseModel):
    # ... existing fields ...
    total_candidates_considered: int = Field(
        ..., 
        description="Total candidates considered before filtering"
    )
```

**Impact:** Better transparency for debugging and analytics

---

## Code Changes Required

### For Rank Field (Priority 1)

**File 1: `api/schemas/models.py`**
```python
# Line 228-240: Update MatchResult
class MatchResult(BaseModel):
    rank: int = Field(..., description="Rank of this match (1-based)", ge=1)
    id: str = Field(..., description="Match ID")
    # ... rest unchanged ...
```

**File 2: `api/routes/upload.py`**
```python
# Line 87-140: Update format_match_results()
def format_match_results(
    matches: List[dict],
    confidence_scoring,
    min_confidence_threshold: float = 0.50
) -> List[MatchResult]:
    results = []
    for rank, match in enumerate(matches, 1):  # ← Add enumerate
        results.append(MatchResult(
            rank=rank,  # ← Add rank
            id=match.get('id', ''),
            # ... rest unchanged ...
        ))
    return results
```

**File 3: `api/routes/search.py`**
```python
# Line 228: Update format_match_results call
matches = format_match_results(found_matches, confidence_scoring)
# No change needed if format_match_results already handles rank
```

---

## Conclusion

The Top-K ranking implementation is **production-ready** and well-designed. The system correctly:
- ✅ Sorts results by combined similarity score
- ✅ Filters candidates by multiple thresholds
- ✅ Returns top-K matches for human verification
- ✅ Supports multi-image aggregation
- ✅ Provides detailed confidence scores

**Minor improvements** (rank field, pagination) would enhance UX but are not critical for functionality.

**Overall Assessment:** ✅ **GOOD** - Ready for production use.

---

**Report Generated:** November 27, 2025  
**Investigator:** AI Assistant  
**Codebase Version:** Phase 3 (Multi-Image Feature Complete)

