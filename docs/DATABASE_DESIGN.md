# Database Design

## 1. Overview

The Missing Person AI system uses **Qdrant** as the primary vector database to store face embeddings and metadata. Qdrant is optimized for similarity search using cosine distance, which is ideal for face recognition tasks.

### Database Technology
- **Vector Database**: Qdrant (v1.12.0+)
- **Distance Metric**: Cosine similarity
- **Vector Dimension**: 512 (ArcFace/InsightFace embeddings)
- **Storage Model**: Hybrid (vectors + JSON payload)

---

## 2. Collections

The system maintains two main collections:

### 2.1 `missing_persons`
Stores profiles of missing persons. Each point represents one image of a person.

### 2.2 `found_persons`
Stores profiles of found persons. Each point represents one image of a person.

**Note**: Multiple images per person are stored as separate points, grouped by `case_id` (missing) or `found_id` (found).

---

## 3. Schema Design

### 3.1 Missing Person Point Structure

**Vector Field**:
- **Type**: `float32[512]`
- **Description**: Face embedding vector extracted by InsightFace ArcFace model
- **Special Case**: Zero vector `[0.0, ..., 0.0]` for reference-only images (no face detected or low quality)

**Payload Fields** (JSON):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | Yes | Unique case identifier (e.g., "MISS_2023_001") |
| `image_id` | string | Yes | Unique image identifier (e.g., "MISS_2023_001_img_0") |
| `image_index` | integer | Yes | Order in upload batch (0-based) |
| `total_images` | integer | Yes | Total number of images for this person |
| `name` | string | Yes | Full name of missing person |
| `age_at_disappearance` | integer | Yes | Age when person disappeared (0-120) |
| `year_disappeared` | integer | Yes | Year of disappearance (1900-2100) |
| `gender` | string | Yes | Gender ("male" or "female") |
| `location_last_seen` | string | Yes | Last known location |
| `contact` | string | Yes | Contact information (family/guardian) |
| `height_cm` | integer | No | Height in centimeters (50-250) |
| `birthmarks` | array[string] | No | List of birthmarks/scars/features |
| `additional_info` | string | No | Additional notes (max 1000 chars) |
| `age_at_photo` | integer | No | Estimated age when photo was taken |
| `photo_year` | integer | No | Year photo was taken |
| `photo_quality_score` | float | No | Face quality score (0.0-1.0) |
| `image_url` | string | No | Cloudinary URL (if uploaded) |
| `is_valid_for_matching` | boolean | Yes | Whether image is used for matching (false for reference-only) |
| `validation_status` | string | Yes | Status: "valid", "no_face_detected", "low_quality" |
| `validation_details` | object | Yes | Detailed validation info (face_detected, quality_score, reason, etc.) |
| `collection_type` | string | Yes | Always "missing" |
| `upload_timestamp` | string | Yes | ISO 8601 timestamp |
| `point_id` | string | Yes | Unique Qdrant point ID (UUID) |
| `estimated_current_age` | integer | Yes | Calculated: `age_at_disappearance + (current_year - year_disappeared)` |

---

### 3.2 Found Person Point Structure

**Vector Field**:
- **Type**: `float32[512]`
- **Description**: Face embedding vector (or zero vector for reference-only images)

**Payload Fields** (JSON):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `found_id` | string | Yes | Unique found person identifier (e.g., "FOUND_2023_001") |
| `case_id` | string | Yes | Alias for found_id (for compatibility) |
| `image_id` | string | Yes | Unique image identifier |
| `image_index` | integer | Yes | Order in upload batch |
| `total_images` | integer | Yes | Total number of images |
| `name` | string | No | Name if known |
| `current_age_estimate` | integer | Yes | Estimated current age (0-120) |
| `gender` | string | Yes | Gender ("male" or "female") |
| `current_location` | string | Yes | Location where person was found |
| `finder_contact` | string | Yes | Contact information of finder |
| `visible_marks` | array[string] | No | List of visible marks/scars |
| `current_condition` | string | No | Current condition/status |
| `additional_info` | string | No | Additional notes |
| `age_at_photo` | integer | No | Estimated age in photo |
| `photo_year` | integer | No | Year photo was taken |
| `photo_quality_score` | float | No | Face quality score (0.0-1.0) |
| `image_url` | string | No | Cloudinary URL |
| `is_valid_for_matching` | boolean | Yes | Whether image is used for matching |
| `validation_status` | string | Yes | Validation status |
| `validation_details` | object | Yes | Detailed validation info |
| `collection_type` | string | Yes | Always "found" |
| `upload_timestamp` | string | Yes | ISO 8601 timestamp |
| `point_id` | string | Yes | Unique Qdrant point ID (UUID) |

---

## 4. Entity Relationship (Conceptual)

```
┌─────────────────────┐
│  Missing Person     │
│  (case_id)          │
└──────────┬──────────┘
           │
           │ 1:N
           │
           ▼
┌─────────────────────┐
│  Missing Images      │
│  (point_id,          │
│   image_id,          │
│   embedding[512],   │
│   metadata)         │
└─────────────────────┘

┌─────────────────────┐
│  Found Person        │
│  (found_id)          │
└──────────┬──────────┘
           │
           │ 1:N
           │
           ▼
┌─────────────────────┐
│  Found Images        │
│  (point_id,          │
│   image_id,          │
│   embedding[512],    │
│   metadata)          │
└─────────────────────┘
```

**Relationship**:
- One person (missing or found) can have multiple images (1-10 per person)
- Each image is stored as a separate point in Qdrant
- Images are grouped by `case_id` (missing) or `found_id` (found)
- Images can be "valid for matching" (with embedding) or "reference-only" (zero vector)

---

## 5. Indexing and Search Strategy

### 5.1 Vector Index
- **Type**: HNSW (Hierarchical Navigable Small World) - default in Qdrant
- **Distance**: Cosine similarity
- **Purpose**: Fast approximate nearest neighbor search for face embeddings

### 5.2 Metadata Filtering
Qdrant supports filtering by payload fields:
- **Gender filter**: Exact match (`gender = "male"` or `"female"`)
- **Age range filter**: Range query on `estimated_current_age` (missing) or `current_age_estimate` (found)
- **Validation filter**: `is_valid_for_matching = true` (only search valid images)

### 5.3 Search Flow
1. **Top-K Vector Search**: Query Qdrant with embedding vector, retrieve top-K nearest neighbors (no score threshold in pure Top-K mode)
2. **Metadata Reranking**: Combine face similarity with metadata scores (age consistency, gender match, location, features)
3. **Final Ranking**: Sort by `combined_score = 0.7 * face_similarity + 0.3 * metadata_similarity`
4. **Return Top-K**: Return best K matches to user

---

## 6. Data Flow

### 6.1 Insert Flow
```
User Upload → API → Face Detection → Embedding Extraction → Qdrant Insert
                ↓
         (Metadata + Vector)
```

### 6.2 Search Flow
```
User Query → API → Get Profile Embedding → Qdrant Vector Search → Rerank → Top-K Results
```

---

## 7. Data Integrity

### 7.1 Constraints
- **Required Fields**: Enforced at API level (Pydantic validation)
- **Vector Dimension**: Must be exactly 512 (validated before insert)
- **Unique Identifiers**: `case_id`/`found_id` + `image_index` combination should be unique per person

### 7.2 Data Quality
- **Face Detection**: Images without faces are marked `is_valid_for_matching = false`
- **Quality Threshold**: Images with quality < 0.60 are marked as reference-only
- **Reference Images**: Stored with zero vector, not used in matching but available for display

---

## 8. Scalability Considerations

- **Horizontal Scaling**: Qdrant supports distributed clusters
- **Collection Sharding**: Can partition by region or time period if needed
- **Vector Compression**: Qdrant supports quantization for memory efficiency
- **Batch Operations**: `insert_batch()` used for multi-image uploads (1-10 images per person)

---

## 9. Backup and Recovery

- **Qdrant Snapshots**: Regular snapshots recommended for production
- **Metadata Backup**: Export payload data separately if needed
- **Image Storage**: Images stored in Cloudinary (external), Qdrant only stores URLs

---

## 10. Example Queries

### 10.1 Insert Missing Person
```python
point = PointStruct(
    id=uuid4(),
    vector=embedding.tolist(),  # 512-D
    payload={
        "case_id": "MISS_2023_001",
        "name": "John Doe",
        "age_at_disappearance": 25,
        "year_disappeared": 2020,
        "gender": "male",
        # ... other fields
    }
)
```

### 10.2 Search Similar Faces
```python
results = qdrant_client.search(
    collection_name="found_persons",
    query_vector=query_embedding,
    limit=10,  # Top-K
    query_filter=Filter(
        must=[
            FieldCondition(key="is_valid_for_matching", match=MatchValue(value=True)),
            FieldCondition(key="gender", match=MatchValue(value="male"))
        ]
    )
)
```

---

## 11. Summary

The database design leverages Qdrant's vector search capabilities to enable fast face similarity matching. The hybrid storage model (vectors + JSON payload) allows both efficient similarity search and flexible metadata querying. The multi-image support (1-10 images per person) improves matching accuracy across age gaps.

