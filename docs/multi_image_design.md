# Multi-Image Profile Feature Design Document

> **Mục đích**: Cho phép mỗi người (missing/found) có nhiều ảnh với metadata tuổi, giúp cải thiện matching khi age gap lớn (>40 năm → similarity chỉ ~0.23 với single image).

---

## 📊 VẤN ĐỀ HIỆN TẠI

### Kết quả phân tích Age Gap vs Similarity (FGNET dataset, 2501 cặp)

| Khoảng cách tuổi | Số cặp | Mean Similarity | Median | Min   | Max   |
|------------------|--------|-----------------|--------|-------|-------|
| 6–10 năm         | 245    | 0.4515          | 0.4537 | 0.06  | 0.92  |
| 11–20 năm        | 1186   | 0.3543          | 0.3454 | -0.05 | 0.85  |
| 21–30 năm        | 674    | 0.2791          | 0.2704 | -0.08 | 0.77  |
| 31–40 năm        | 319    | 0.2451          | 0.2355 | -0.07 | 0.69  |
| 41+ năm          | 77     | 0.2291          | 0.2161 | -0.01 | 0.56  |

**Kết luận**: Khi age gap > 30 năm, mean similarity < 0.25 → dưới ngưỡng 0.30 → **bỏ lỡ match**.

---

## 🏗️ PHASE 1: CURRENT ARCHITECTURE ANALYSIS

### 1. Storage Architecture

```
Upload → Read bytes → Normalize orientation → Enhance quality 
      → MTCNN Detect face → Align → InsightFace Extract 512-D embedding 
      → Upload to Cloudinary (optional) → Store in Qdrant
```

**Key files:**
- `api/routes/upload.py`: Main upload endpoints
- `services/vector_db.py`: Qdrant operations
- `models/face_detection.py`: MTCNN detection + alignment
- `models/face_embedding_insightface.py`: 512-D ArcFace embedding

### 2. Qdrant Current State

**Collections:**
- `missing_persons`: Người mất tích
- `found_persons`: Người được tìm thấy

**Current Point Structure (1 point = 1 person = 1 image):**
```python
{
    "id": "MISS_xxxx_yyyyMMdd_HHmmss",  # case_id
    "vector": [512-D float array],
    "payload": {
        "case_id": "MISS_xxxx_yyyyMMdd_HHmmss",
        "name": "Nguyen Van A",
        "age_at_disappearance": 25,
        "year_disappeared": 2020,
        "gender": "male",
        "last_seen_location": "Ha Noi",
        "distinctive_features": "Scar on left cheek",
        "contact_info": "0901234567",
        "additional_info": "...",
        "image_url": "https://cloudinary.com/...",
        "created_at": "2024-01-15T10:30:00"
    }
}
```

### 3. API Endpoints

| Endpoint | Method | Chức năng |
|----------|--------|-----------|
| `/api/v1/upload/missing` | POST | Upload missing person (1 ảnh) |
| `/api/v1/upload/found` | POST | Upload found person (1 ảnh) |
| `/api/v1/search/missing/{case_id}` | GET | Get missing person + matches |
| `/api/v1/search/found/{found_id}` | GET | Get found person + matches |
| `/api/v1/search/cases/all` | GET | List all cases |

### 4. Bilateral Search Flow

```python
# services/bilateral_search.py
1. Upload missing → search found_persons collection
2. Upload found → search missing_persons collection
3. Apply age-appropriate threshold:
   - Child (age < 13): threshold = 0.35
   - Adult: threshold = 0.30
4. Rerank by metadata (location, features, etc.)
5. Return top matches with confidence scores
```

### 5. Key Findings

✅ **Strengths:**
- `case_id` đã là unique identifier → có thể dùng làm group key
- Qdrant hỗ trợ filter by payload field
- Pipeline embedding đã modular (detect → align → embed)
- Cloudinary upload đã có sẵn

⚠️ **Bottlenecks:**
- Current: 1 point = 1 person = 1 image
- Không có `image_id` riêng
- Search trả về points, không aggregate theo person

🔴 **Breaking Changes Required:**
- Schema mới cần `image_id`, `image_index`, `age_at_photo`
- Search logic cần aggregate multiple points → 1 person
- API cần hỗ trợ multi-file upload

---

## 🎯 PHASE 2: MULTI-IMAGE DESIGN

### Design Decision: Option A - One Point Per Image

**Lý do chọn:**
- Tận dụng Qdrant vector search có sẵn
- Không cần custom distance function
- Simple implementation, backward compatible
- Dễ scale (thêm ảnh = thêm point)

### 2.1 NEW QDRANT POINT SCHEMA

```python
{
    # ═══════════════════════════════════════════════════════════════
    # GROUPING & IDENTIFICATION
    # ═══════════════════════════════════════════════════════════════
    "case_id": "MISS_001",              # Primary group key (unchanged)
    "image_id": "MISS_001_img_0",       # Unique per image
    "image_index": 0,                    # 0, 1, 2, ... (0 = primary)
    "total_images": 3,                   # Total images for this person
    
    # ═══════════════════════════════════════════════════════════════
    # PER-IMAGE METADATA (unique to each point)
    # ═══════════════════════════════════════════════════════════════
    "age_at_photo": 8,                   # REQUIRED for multi-image benefit
    "photo_year": 2010,                  # Year photo was taken
    "photo_quality_score": 0.85,         # MTCNN confidence
    "is_primary": true,                  # Display preference (image_index=0)
    "image_url": "https://...",          # Cloudinary URL for this image
    
    # ═══════════════════════════════════════════════════════════════
    # SHARED PERSON METADATA (duplicated across all images)
    # ═══════════════════════════════════════════════════════════════
    "name": "Nguyen Van A",
    "age_at_disappearance": 25,
    "year_disappeared": 2020,
    "gender": "male",
    "last_seen_location": "Ha Noi",
    "distinctive_features": "Scar on left cheek",
    "contact_info": "0901234567",
    "additional_info": "...",
    "created_at": "2024-01-15T10:30:00"
}
```

### 2.2 API CHANGES

#### New Upload Endpoint (Multi-Image)

```python
# POST /api/v1/upload/missing/multi
@router.post("/missing/multi", response_model=MultiUploadResponse)
async def upload_missing_person_multi(
    files: List[UploadFile] = File(...),           # Multiple images
    ages_at_photos: List[int] = Form(...),         # Age at each photo
    photo_years: Optional[List[int]] = Form(None), # Year of each photo
    
    # Shared metadata (same as before)
    name: str = Form(...),
    age_at_disappearance: int = Form(...),
    year_disappeared: int = Form(...),
    gender: str = Form(...),
    last_seen_location: str = Form(...),
    distinctive_features: Optional[str] = Form(None),
    contact_info: str = Form(...),
    additional_info: Optional[str] = Form(None),
):
    # Validation
    if len(files) != len(ages_at_photos):
        raise HTTPException(400, "Number of files must match ages_at_photos")
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images per person")
    
    # Process each image
    case_id = generate_case_id()
    points = []
    
    for idx, (file, age) in enumerate(zip(files, ages_at_photos)):
        # Detect + Embed
        face = detector.extract_largest_face(image)
        embedding = embedder.extract_embedding(face)
        
        # Upload to Cloudinary
        image_url = upload_to_cloudinary(file, f"{case_id}_img_{idx}")
        
        # Create point
        point = {
            "id": f"{case_id}_img_{idx}",
            "vector": embedding,
            "payload": {
                "case_id": case_id,
                "image_id": f"{case_id}_img_{idx}",
                "image_index": idx,
                "total_images": len(files),
                "age_at_photo": age,
                "photo_year": photo_years[idx] if photo_years else None,
                "photo_quality_score": face_confidence,
                "is_primary": (idx == 0),
                "image_url": image_url,
                # Shared metadata
                "name": name,
                "age_at_disappearance": age_at_disappearance,
                # ... rest of metadata
            }
        }
        points.append(point)
    
    # Batch insert to Qdrant
    qdrant.upsert(collection="missing_persons", points=points)
    
    # Trigger bilateral search
    matches = bilateral_search.search_multi_image(case_id, points)
    
    return MultiUploadResponse(case_id=case_id, images_uploaded=len(files), matches=matches)
```

#### Backward Compatible: Keep Single-Image Endpoint

```python
# POST /api/v1/upload/missing (unchanged, sets total_images=1)
```

### 2.3 SEARCH AGGREGATION STRATEGY

#### Best Match Per Person (Recommended)

```python
def search_multi_image(query_case_id: str, query_points: List[dict]) -> List[MatchResult]:
    """
    Search strategy for multi-image query:
    1. Query Qdrant with EACH query embedding
    2. Aggregate results by target case_id
    3. Use BEST similarity score per person pair
    4. Apply age-aware threshold
    """
    
    all_results = {}  # {target_case_id: best_match_info}
    
    for query_point in query_points:
        query_embedding = query_point["vector"]
        query_age = query_point["payload"]["age_at_photo"]
        
        # Search Qdrant
        hits = qdrant.search(
            collection="found_persons",
            query_vector=query_embedding,
            limit=100,  # Get more results for aggregation
            with_payload=True
        )
        
        for hit in hits:
            target_case_id = hit.payload["case_id"]
            target_age = hit.payload["age_at_photo"]
            similarity = hit.score
            
            # Calculate age gap for this image pair
            age_gap = abs(query_age - target_age)
            
            # Track best match per target person
            if target_case_id not in all_results:
                all_results[target_case_id] = {
                    "best_similarity": similarity,
                    "best_age_gap": age_gap,
                    "query_image_id": query_point["payload"]["image_id"],
                    "target_image_id": hit.payload["image_id"],
                    "payload": hit.payload
                }
            else:
                # Update if this pair has BETTER similarity
                if similarity > all_results[target_case_id]["best_similarity"]:
                    all_results[target_case_id] = {
                        "best_similarity": similarity,
                        "best_age_gap": age_gap,
                        "query_image_id": query_point["payload"]["image_id"],
                        "target_image_id": hit.payload["image_id"],
                        "payload": hit.payload
                    }
    
    # Filter by threshold and sort
    matches = []
    for case_id, info in all_results.items():
        threshold = get_age_aware_threshold(info["best_age_gap"])
        if info["best_similarity"] >= threshold:
            matches.append(MatchResult(
                case_id=case_id,
                similarity=info["best_similarity"],
                age_gap=info["best_age_gap"],
                matched_query_image=info["query_image_id"],
                matched_target_image=info["target_image_id"],
                # ... other fields from payload
            ))
    
    # Sort by similarity descending
    matches.sort(key=lambda x: x.similarity, reverse=True)
    return matches[:20]  # Top 20


def get_age_aware_threshold(age_gap: int) -> float:
    """
    Dynamic threshold based on age gap between matched photos.
    Closer ages → stricter threshold (expect higher similarity)
    Larger gap → relaxed threshold (expect lower similarity)
    """
    if age_gap <= 5:
        return 0.35  # Strict: same age range should match well
    elif age_gap <= 10:
        return 0.30  # Normal
    elif age_gap <= 20:
        return 0.25  # Relaxed
    elif age_gap <= 30:
        return 0.22  # More relaxed
    else:
        return 0.18  # Very relaxed for 30+ year gap
```

### 2.4 NEW PYDANTIC SCHEMAS

```python
# api/schemas/models.py

class ImageMetadata(BaseModel):
    """Metadata for a single image in multi-image upload"""
    age_at_photo: int = Field(..., ge=0, le=120, description="Age when photo was taken")
    photo_year: Optional[int] = Field(None, ge=1900, le=2100)


class MultiImageUploadRequest(BaseModel):
    """Request body for multi-image upload (metadata part)"""
    name: str
    age_at_disappearance: int = Field(..., ge=0, le=120)
    year_disappeared: int = Field(..., ge=1900, le=2100)
    gender: str
    last_seen_location: str
    distinctive_features: Optional[str] = None
    contact_info: str
    additional_info: Optional[str] = None
    
    # Per-image metadata
    images_metadata: List[ImageMetadata]


class MultiUploadResponse(BaseModel):
    """Response for multi-image upload"""
    success: bool
    case_id: str
    images_uploaded: int
    images_failed: int = 0
    failed_reasons: List[str] = []
    matches: List[MatchResult] = []


class ImageInfo(BaseModel):
    """Info about a single image in a person's profile"""
    image_id: str
    image_index: int
    image_url: str
    age_at_photo: int
    photo_year: Optional[int]
    photo_quality_score: float
    is_primary: bool


class PersonProfileResponse(BaseModel):
    """Full person profile with all images"""
    case_id: str
    name: str
    age_at_disappearance: int
    year_disappeared: int
    gender: str
    last_seen_location: str
    distinctive_features: Optional[str]
    contact_info: str
    additional_info: Optional[str]
    created_at: datetime
    
    # Multi-image
    total_images: int
    images: List[ImageInfo]
    
    # Matches
    matches: List[MatchResult] = []
```

### 2.5 VECTOR_DB SERVICE CHANGES

```python
# services/vector_db.py

class VectorDatabaseService:
    
    def insert_multi_image_person(
        self,
        collection: str,
        case_id: str,
        embeddings: List[np.ndarray],
        images_metadata: List[dict],
        shared_metadata: dict
    ) -> bool:
        """Insert multiple points for one person"""
        
        points = []
        for idx, (embedding, img_meta) in enumerate(zip(embeddings, images_metadata)):
            point = PointStruct(
                id=f"{case_id}_img_{idx}",
                vector=embedding.tolist(),
                payload={
                    "case_id": case_id,
                    "image_id": f"{case_id}_img_{idx}",
                    "image_index": idx,
                    "total_images": len(embeddings),
                    "is_primary": (idx == 0),
                    **img_meta,      # age_at_photo, photo_year, image_url, etc.
                    **shared_metadata # name, gender, etc.
                }
            )
            points.append(point)
        
        self.client.upsert(collection_name=collection, points=points)
        return True
    
    def get_person_all_images(self, collection: str, case_id: str) -> List[dict]:
        """Get all images/points for a person"""
        
        results = self.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
            ),
            with_payload=True,
            with_vectors=True
        )
        return [{"id": p.id, "vector": p.vector, "payload": p.payload} for p in results[0]]
    
    def delete_person(self, collection: str, case_id: str) -> bool:
        """Delete ALL images for a person"""
        
        self.client.delete(
            collection_name=collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
                )
            )
        )
        return True
    
    def add_image_to_person(
        self,
        collection: str,
        case_id: str,
        embedding: np.ndarray,
        image_metadata: dict
    ) -> str:
        """Add a new image to existing person"""
        
        # Get current images to determine next index
        existing = self.get_person_all_images(collection, case_id)
        if not existing:
            raise ValueError(f"Person {case_id} not found")
        
        next_index = len(existing)
        new_total = next_index + 1
        image_id = f"{case_id}_img_{next_index}"
        
        # Get shared metadata from existing point
        shared_metadata = {k: v for k, v in existing[0]["payload"].items() 
                          if k not in ["image_id", "image_index", "total_images", 
                                       "age_at_photo", "photo_year", "image_url",
                                       "photo_quality_score", "is_primary"]}
        
        # Insert new point
        point = PointStruct(
            id=image_id,
            vector=embedding.tolist(),
            payload={
                "case_id": case_id,
                "image_id": image_id,
                "image_index": next_index,
                "total_images": new_total,
                "is_primary": False,
                **image_metadata,
                **shared_metadata
            }
        )
        self.client.upsert(collection_name=collection, points=[point])
        
        # Update total_images in all existing points
        for p in existing:
            self.client.set_payload(
                collection_name=collection,
                payload={"total_images": new_total},
                points=[p["id"]]
            )
        
        return image_id
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Schema & Core (Priority: HIGH)

- [ ] **1.1** Update `api/schemas/models.py` with new Pydantic models
- [ ] **1.2** Update `services/vector_db.py` with multi-image methods
- [ ] **1.3** Create migration script for existing data (add `image_id`, `image_index=0`, `total_images=1`)

### Phase 2: Upload Flow (Priority: HIGH)

- [ ] **2.1** Add `POST /api/v1/upload/missing/multi` endpoint
- [ ] **2.2** Add `POST /api/v1/upload/found/multi` endpoint
- [ ] **2.3** Add `POST /api/v1/upload/{case_id}/add-image` endpoint
- [ ] **2.4** Update validation for multi-file upload

### Phase 3: Search & Aggregation (Priority: HIGH)

- [ ] **3.1** Implement `search_multi_image()` in `bilateral_search.py`
- [ ] **3.2** Implement age-aware dynamic thresholds
- [ ] **3.3** Update `GET /api/v1/search/missing/{case_id}` to return all images

### Phase 4: Testing (Priority: MEDIUM)

- [ ] **4.1** Unit tests for multi-image upload
- [ ] **4.2** Unit tests for search aggregation
- [ ] **4.3** Integration test: upload 3 images → search → verify best match

### Phase 5: Frontend Support (Priority: LOW)

- [ ] **5.1** API documentation update
- [ ] **5.2** Example requests in Postman/Swagger

---

## 🔧 MIGRATION SCRIPT

```python
# scripts/migrate_to_multi_image.py
"""
Migrate existing single-image data to multi-image schema.
Safe to run multiple times (idempotent).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

def migrate_collection(client: QdrantClient, collection: str):
    """Add multi-image fields to existing points"""
    
    # Scroll through all points
    offset = None
    migrated = 0
    
    while True:
        results, offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=100,
            with_payload=True
        )
        
        if not results:
            break
        
        for point in results:
            payload = point.payload
            
            # Skip if already migrated
            if "image_id" in payload:
                continue
            
            case_id = payload.get("case_id", point.id)
            
            # Add new fields
            new_fields = {
                "image_id": f"{case_id}_img_0",
                "image_index": 0,
                "total_images": 1,
                "is_primary": True,
                "age_at_photo": payload.get("age_at_disappearance"),  # Default to disappearance age
                "photo_quality_score": 0.0  # Unknown for old data
            }
            
            client.set_payload(
                collection_name=collection,
                payload=new_fields,
                points=[point.id]
            )
            migrated += 1
        
        if offset is None:
            break
    
    print(f"Migrated {migrated} points in {collection}")


if __name__ == "__main__":
    client = QdrantClient(host="localhost", port=6333)
    
    migrate_collection(client, "missing_persons")
    migrate_collection(client, "found_persons")
    
    print("Migration complete!")
```

---

## 📊 PERFORMANCE ESTIMATES

| Metric | Single-Image (Current) | Multi-Image (5 avg) |
|--------|------------------------|---------------------|
| Points per person | 1 | 5 |
| Storage per person | ~6KB | ~30KB |
| Search queries per upload | 1 | 5 |
| Search latency | ~50ms | ~150ms |
| Match accuracy (40+ year gap) | ~23% | ~60-70% (estimated) |

---

## 🚀 QUICK START FOR NEW CHAT

Trong chat mới, bạn có thể nói:

> "Đọc file `BE/docs/multi_image_design.md` và bắt đầu implement Phase 1: Schema & Core. 
> Cụ thể: update `api/schemas/models.py` với các Pydantic models mới."

Hoặc:

> "Implement multi-image feature theo design trong `BE/docs/multi_image_design.md`. 
> Bắt đầu từ checklist item 1.1."

