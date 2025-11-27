"""
Pydantic models for Missing Person AI API.

This module defines request and response schemas for API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class GenderEnum(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"


class ConfidenceLevelEnum(str, Enum):
    """Confidence level enumeration."""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


# Request Models

class MissingPersonMetadata(BaseModel):
    """Missing person metadata schema."""
    case_id: Optional[str] = Field(None, description="Unique case identifier (auto-generated if not provided)", example="MISS_2023_001")
    name: str = Field(..., min_length=2, max_length=100, description="Full name", example="John Doe")
    age_at_disappearance: int = Field(..., ge=0, le=120, description="Age when disappeared", example=25)
    year_disappeared: int = Field(..., ge=1900, le=2024, description="Year of disappearance", example=2020)
    gender: GenderEnum = Field(..., description="Gender", example="male")
    location_last_seen: str = Field(..., min_length=3, max_length=200, description="Last known location", example="New York, NY")
    contact: str = Field(..., description="Contact information", example="family@example.com")
    height_cm: Optional[int] = Field(None, ge=50, le=250, description="Height in centimeters", example=175)
    birthmarks: Optional[List[str]] = Field(None, max_items=10, description="List of birthmarks/scars", example=["scar on left arm"])
    additional_info: Optional[str] = Field(None, max_length=1000, description="Additional information")

    @validator('birthmarks')
    def validate_birthmarks(cls, v):
        if v is not None:
            return [mark.strip() for mark in v if mark.strip()]
        return v


class FoundPersonMetadata(BaseModel):
    """Found person metadata schema."""
    found_id: Optional[str] = Field(None, description="Unique found person identifier (auto-generated if not provided)", example="FOUND_2023_001")
    name: Optional[str] = Field(None, min_length=2, max_length=100, description="Name of the found person (optional)", example="John Doe")
    current_age_estimate: int = Field(..., ge=0, le=120, description="Estimated current age", example=30)
    gender: GenderEnum = Field(..., description="Gender", example="male")
    current_location: str = Field(..., min_length=3, max_length=200, description="Current location", example="Los Angeles, CA")
    finder_contact: str = Field(..., description="Finder contact information", example="finder@example.com")
    visible_marks: Optional[List[str]] = Field(None, max_items=10, description="List of visible marks/scars", example=["scar on left arm"])
    current_condition: Optional[str] = Field(None, max_length=500, description="Current condition/status")
    additional_info: Optional[str] = Field(None, max_length=1000, description="Additional information")

    @validator('visible_marks')
    def validate_visible_marks(cls, v):
        if v is not None:
            return [mark.strip() for mark in v if mark.strip()]
        return v


class SearchFilters(BaseModel):
    """Search filters schema."""
    gender: Optional[GenderEnum] = Field(None, description="Gender filter")
    age_range: Optional[List[int]] = Field(None, min_items=2, max_items=2, description="Age range [min, max]", example=[20, 40])
    location: Optional[str] = Field(None, description="Location filter")

    @validator('age_range')
    def validate_age_range(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("Age range must have exactly 2 values")
            if v[0] > v[1]:
                raise ValueError("Minimum age must be less than maximum age")
            if not all(0 <= age <= 120 for age in v):
                raise ValueError("Ages must be between 0 and 120")
        return v


class SearchParameters(BaseModel):
    """Search parameters schema."""
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Similarity threshold")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")


# Response Models

class FaceQualityMetrics(BaseModel):
    """Face quality metrics schema."""
    sharpness: float = Field(..., description="Sharpness score (Laplacian variance)")
    brightness: float = Field(..., description="Brightness score (mean pixel value)")
    contrast: float = Field(..., description="Contrast score (standard deviation)")
    is_sharp: bool = Field(..., description="Whether face is sharp enough")
    is_bright_enough: bool = Field(..., description="Whether face has good brightness")
    is_contrasted: bool = Field(..., description="Whether face has good contrast")
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized quality score (0-1 scale) aggregated from sharpness/brightness/contrast"
    )


class ConfidenceFactor(BaseModel):
    """Confidence factor schema."""
    score: float = Field(..., ge=0.0, le=1.0, description="Factor score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Factor weight")
    contribution: float = Field(..., ge=0.0, le=1.0, description="Factor contribution to overall score")
    description: str = Field(..., description="Human-readable description")


class ConfidenceExplanation(BaseModel):
    """Confidence explanation schema."""
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    factors: Dict[str, ConfidenceFactor] = Field(..., description="Individual factor scores")
    reasons: List[str] = Field(..., description="List of reasons supporting the match")
    summary: str = Field(..., description="Overall summary of the match")
    recommendations: List[str] = Field(..., description="Recommended next steps")
    threshold_info: Dict[str, float] = Field(..., description="Confidence level thresholds")
    multi_image_details: Optional['MultiImageMatchDetails'] = Field(
        None, 
        description="Multi-image matching details (only present for multi-image matches)"
    )


# ============================================================================
# Multi-Image Upload Schemas
# ============================================================================

class UploadedImageInfo(BaseModel):
    """Information about a valid uploaded image (used for matching)."""
    image_id: str = Field(..., description="Unique image identifier", example="MISS_001_img_0")
    image_index: int = Field(..., ge=0, description="Index of image in upload batch (0-based)", example=0)
    image_url: Optional[str] = Field(None, description="Cloudinary URL for the image")
    age_at_photo: Optional[int] = Field(None, ge=0, le=120, description="Age when photo was taken (None if photo_year not provided)", example=8)
    photo_year: Optional[int] = Field(None, description="Year photo was taken", example=2010)
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Photo quality score", example=0.85)
    validation_status: str = Field(default="valid", description="Always 'valid' for this type")


class ReferenceImageInfo(BaseModel):
    """Information about a reference-only image (not used for matching)."""
    image_id: str = Field(..., description="Unique image identifier", example="MISS_001_img_1")
    image_index: int = Field(..., ge=0, description="Index in upload batch", example=1)
    image_url: Optional[str] = Field(None, description="Cloudinary URL for the image")
    age_at_photo: Optional[int] = Field(None, ge=0, le=120, description="Estimated age when photo taken (None if photo_year not provided)", example=10)
    photo_year: Optional[int] = Field(None, description="Year photo was taken if known", example=2005)
    validation_status: str = Field(
        ..., 
        description="Reason: 'no_face_detected', 'low_quality', 'face_too_small', 'multiple_faces'",
        example="no_face_detected"
    )
    reason: str = Field(
        ..., 
        description="Human-readable explanation",
        example="MTCNN could not detect face. Image saved for reference purposes."
    )


class FailedImageInfo(BaseModel):
    """Information about a failed image upload."""
    filename: str = Field(..., description="Original filename of the failed image")
    index: int = Field(..., ge=0, description="Index in the upload batch")
    reason: str = Field(..., description="Reason for failure", example="No face detected")


class MultiImageUploadResponse(BaseModel):
    """Response for batch/multi-image upload."""
    success: bool = Field(..., description="Overall upload success status")
    message: str = Field(..., description="Human-readable message")
    case_id: Optional[str] = Field(None, description="Case ID or Found ID for the person", example="MISS_001")
    total_images_uploaded: int = Field(..., ge=0, description="Total images saved (valid + reference)")
    total_images_failed: int = Field(default=0, ge=0, description="Number of failed images (file errors)")
    
    # Separate valid and reference images
    valid_images: List[UploadedImageInfo] = Field(
        default=[], 
        description="Images used for matching (with face embeddings)"
    )
    reference_images: List[ReferenceImageInfo] = Field(
        default=[], 
        description="Images saved for reference only (no face detected or low quality)"
    )
    failed_images: List[FailedImageInfo] = Field(
        default=[], 
        description="Images that failed to upload (file errors)"
    )
    
    # Counts for quick access
    matching_images_count: int = Field(..., description="Number of images used for matching")
    reference_images_count: int = Field(default=0, description="Number of reference-only images")
    
    potential_matches: List['MatchResult'] = Field(default=[], description="Potential matches found")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class MultiImageMatchDetails(BaseModel):
    """Detailed information about multi-image matching."""
    total_query_images: int = Field(..., ge=1, description="Number of query images used")
    total_candidate_images: int = Field(..., ge=1, description="Number of candidate images")
    num_comparisons: int = Field(..., ge=1, description="Total pairwise comparisons performed")
    best_similarity: float = Field(..., ge=0.0, le=1.0, description="Best pairwise similarity score")
    mean_similarity: float = Field(..., ge=0.0, le=1.0, description="Mean of top similarities")
    consistency_score: float = Field(..., ge=0.0, le=1.0, description="Consistency across image pairs")
    best_match_pair: Dict[str, int] = Field(
        ..., 
        description="Indices of best matching image pair",
        example={"query_idx": 2, "candidate_idx": 1}
    )
    query_age_at_best_match: int = Field(..., description="Query person's age in best matching photo")
    candidate_age_at_best_match: int = Field(..., description="Candidate person's age in best matching photo")
    age_gap_at_best_match: int = Field(..., ge=0, description="Age gap between best matching photos")
    age_bracket_match_found: bool = Field(..., description="Whether similar-age photos were found")
    num_good_matches: int = Field(..., ge=0, description="Number of pairs above good match threshold")


class PersonRecord(BaseModel):
    """Person record schema (for the queried person, not a match)."""
    id: str = Field(..., description="Record ID")
    contact: str = Field(..., description="Contact information")
    metadata: Dict[str, Any] = Field(..., description="Full metadata of the record")
    image_url: Optional[str] = Field(None, description="Cloudinary URL of the person's image")


class MatchResult(BaseModel):
    """Match result schema."""
    id: str = Field(..., description="Match ID")
    face_similarity: float = Field(..., ge=0.0, le=1.0, description="Face similarity score")
    metadata_similarity: float = Field(..., ge=0.0, le=1.0, description="Metadata similarity score")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Combined similarity score")
    confidence_level: ConfidenceLevelEnum = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: ConfidenceExplanation = Field(..., description="Detailed confidence explanation")
    contact: str = Field(..., description="Contact information for this match")
    metadata: Dict[str, Any] = Field(..., description="Full metadata of the matched record")
    image_url: Optional[str] = Field(None, description="Cloudinary URL of the matched person's image")


class UploadResponse(BaseModel):
    """Upload response schema."""
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")
    point_id: Optional[str] = Field(None, description="Database point ID")
    potential_matches: List[MatchResult] = Field(default_factory=list, description="Potential matches found")
    face_quality: Optional[FaceQualityMetrics] = Field(None, description="Face quality assessment")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    case_id: Optional[str] = Field(None, description="Auto-generated case identifier (for missing uploads)")
    found_id: Optional[str] = Field(None, description="Auto-generated found identifier (for found uploads)")
    image_url: Optional[str] = Field(None, description="Cloudinary URL of uploaded image")


class SearchResponse(BaseModel):
    """Search response schema."""
    success: bool = Field(..., description="Whether search was successful")
    message: str = Field(..., description="Response message")
    missing_person: Optional[PersonRecord] = Field(None, description="The missing person record (when searching by missing case_id)")
    found_person: Optional[PersonRecord] = Field(None, description="The found person record (when searching by found_id)")
    matches: List[MatchResult] = Field(default_factory=list, description="Potential matches found in opposite collection (e.g., found_persons when searching missing, or missing_persons when searching found)")
    total_found: int = Field(..., description="Total number of potential matches found")
    search_parameters: SearchParameters = Field(..., description="Search parameters used")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp (Unix time)")
    services: Dict[str, bool] = Field(..., description="Individual service health status")
    database_stats: Optional[Dict[str, Any]] = Field(None, description="Database statistics")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class APIInfo(BaseModel):
    """API information schema."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    documentation_url: str = Field(..., description="Documentation URL")


# Statistics Models

class CollectionStats(BaseModel):
    """Collection statistics schema."""
    collection_name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of points in collection")
    vector_size: int = Field(..., description="Vector dimension")
    last_updated: Optional[float] = Field(None, description="Last update timestamp (Unix time)")


class SystemStats(BaseModel):
    """System statistics schema."""
    missing_persons: CollectionStats = Field(..., description="Missing persons collection stats")
    found_persons: CollectionStats = Field(..., description="Found persons collection stats")
    total_records: int = Field(..., description="Total number of records")
    total_searches_today: Optional[int] = Field(None, description="Number of searches today")
    average_processing_time_ms: Optional[float] = Field(None, description="Average processing time")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")


# List Cases Models

class CaseRecord(BaseModel):
    """Individual case record schema."""
    id: str = Field(..., description="Case ID")
    type: str = Field(..., description="Case type: 'missing' or 'found'")
    name: Optional[str] = Field(None, description="Person name")
    age: Optional[int] = Field(None, description="Person age")
    gender: Optional[str] = Field(None, description="Person gender")
    last_seen_location: Optional[str] = Field(None, description="Last seen location")
    contact: Optional[str] = Field(None, description="Contact information")
    image_url: Optional[str] = Field(None, description="Image URL")
    upload_timestamp: Optional[Union[str, float]] = Field(None, description="Upload timestamp (ISO string or Unix timestamp)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AllCasesResponse(BaseModel):
    """Response schema for listing all cases."""
    success: bool = Field(..., description="Whether request was successful")
    message: str = Field(..., description="Response message")
    cases: List[CaseRecord] = Field(..., description="List of all cases")
    total_count: int = Field(..., description="Total number of cases")
    missing_count: int = Field(..., description="Number of missing persons")
    found_count: int = Field(..., description="Number of found persons")


# Validation Models

class ValidationResult(BaseModel):
    """Validation result schema."""
    is_valid: bool = Field(..., description="Whether input is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")


# Update forward references
MultiImageUploadResponse.model_rebuild()
ConfidenceExplanation.model_rebuild()
